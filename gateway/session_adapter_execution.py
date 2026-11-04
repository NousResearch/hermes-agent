"""Optional adapter-local binding around the actual restored native handler event."""
from copy import deepcopy
from dataclasses import replace

from hermes_state_runtime import RuntimeStoreError


def _binding(event):
    # Only declared data: source dynamic fields can own noncopyable transport objects.
    return deepcopy(replace(event, source=replace(event.source), raw_message=None))


async def execute_native_handler(runner, payload, event):
    """Called inside the authority's execution profile, never instead of admission checks.

    The adapter may bind process-local references/context, not rewrite persisted input.
    A scope cannot suppress a handler failure or cancellation into a successful result.
    """
    envelope = payload.get('native_text_v1')
    if (envelope is None or 'automation' in envelope or event.internal
            or event.source.platform.value == 'api_server'):
        return await runner._handle_message(event)
    provenance = envelope.get('provenance')
    from gateway.session_ingress_context import restore_provenance
    from gateway.session_envelope import _validate_adapter_metadata
    if provenance is not None:
        restore_provenance(runner, event.source, provenance)
    # Execution-profile entry awaited after restore, even for metadata-only adapters.
    _validate_adapter_metadata(runner, event, provenance)
    adapter = runner._adapter_for_source(event.source)
    scope = getattr(type(adapter), 'native_execution_scope', None)
    if scope is None:
        return await runner._handle_message(event)
    event._native_execution_lifecycle = True
    source, before, failure = event.source, _binding(event), None
    async with scope(adapter, event):
        try:
            if event.source is not source or _binding(event) != before:
                raise RuntimeStoreError('admission_conflict')
            if provenance is not None:
                restore_provenance(runner, event.source, provenance)
            if runner._adapter_for_source(event.source) is not adapter:
                raise RuntimeStoreError('not_found')
            _validate_adapter_metadata(runner, event, provenance)
            result = await runner._handle_message(event)
        except BaseException as exc:
            failure = exc
            raise
    if failure is not None:
        raise failure
    return result


def prepare_native_waiter(authority, event):
    """The live input is not the delivery owner until its exact waiter hands back a reply."""
    if event.internal or event.source.platform.value == 'api_server':
        return
    resolve = getattr(getattr(authority, 'runner', None), '_adapter_for_source', None)
    adapter = resolve(event.source) if resolve else None
    if getattr(type(adapter), 'native_execution_scope', None) is not None:
        event._native_completion_owner = False


def handoff_native_execution(actual, original):
    if getattr(actual, '_native_execution_lifecycle', False):
        original._native_execution_event = actual
        actual._native_delivery_owner = original


async def complete_native_execution(authority, adapter, event, admission_id, delivered, home=None):
    """Complete only the opt-in, non-producer lifecycle whose delivery the drain owns."""
    if not getattr(event, '_native_execution_lifecycle', False):
        return
    if getattr(event, '_native_producer_lifecycle', False):
        return
    from gateway.platforms.event import ProcessingOutcome
    from gateway.run_turn_followup_ack import _run_followup_processing_hook
    from gateway.session_ingress import _execution_scope
    from hermes_state_runtime import get_session_admission
    try:
        with await _execution_scope(authority, home):
            row = get_session_admission(authority.db, admission_id=admission_id)
            state = None if row is None else (row['status'], row.get('outcome'))
            outcome = ProcessingOutcome.FAILURE
            if state == ('terminal', 'completed') and delivered:
                outcome = ProcessingOutcome.SUCCESS
            elif state in (('terminal', 'cancelled'), ('terminal', 'interrupted')):
                outcome = ProcessingOutcome.CANCELLED
            await _run_followup_processing_hook(adapter, event, 'on_processing_complete', outcome)
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Native completion failed for admission %s', admission_id)


def claim_native_completion(event):
    if hasattr(event, '_native_completion_owner'):
        event._native_completion_owner = True


def native_admission_refused(event, error, write_started):
    # These transaction guards refuse before INSERT; starting a writer is not a commit.
    refused = error.reason in ('invalid_params', 'queue_full') or (
        error.reason == 'stale_generation' and error.field == 'route')
    if not write_started or (refused and not getattr(event, '_gateway_accepted', False)):
        claim_native_completion(event)
        if hasattr(event, '_shared_busy_admission'):
            event._shared_busy_admission = ('refused', error.reason)
