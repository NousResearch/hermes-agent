"""Local owner-only findings control; shared transports get a content-free pointer."""
import asyncio
from pathlib import Path
import time

from gateway.config import Platform
from gateway.connected_assistance_delivery import owned_local_target
from gateway.session_authorities import active_authority, owner_scope

LOCAL_POINTER = "Open `hermes connectors findings` in your local terminal to review or manage private findings."


def _dispatch(authority, raw):
    from agent.connected_assistance import revalidate_evidence
    from gateway.connected_assistance_store import AssistanceStore
    from hermes_cli.connected_assistance import dispatch_findings_command, parse_findings_args

    with owner_scope(authority, hydrate_secrets=True):
        try:
            args = parse_findings_args(raw)
        except ValueError:
            return dispatch_findings_command(raw, home=Path(authority.profile_id))
        if args.action in {"list", "show"}:
            store = AssistanceStore(Path(authority.profile_id))
            rows = ([store.get_finding(args.identifier)] if args.action == "show" else
                    store.list_findings(status=None if args.status == "all" else args.status, limit=args.limit))
            evidence = [e for row in rows if row for e in row["evidence"]]
            if evidence and not revalidate_evidence(evidence, deadline=time.monotonic() + 10):
                return "Current source access could not be verified. " + LOCAL_POINTER
        return dispatch_findings_command(raw, home=Path(authority.profile_id))


class GatewayFindingsCommandsMixin:
    async def _handle_findings_command(self, event):
        source = event.source
        if source.platform != Platform.LOCAL:
            return LOCAL_POINTER
        authority = active_authority(self)
        if (authority is None or getattr(authority, "retiring", False)
                or getattr(self, "_draining", False)
                or owned_local_target(authority, source.chat_id, source=source) is None):
            return LOCAL_POINTER
        raw = event.text.split(None, 1)
        result = await asyncio.to_thread(_dispatch, authority, raw[1] if len(raw) > 1 else "")
        if (getattr(authority, "retiring", False) or getattr(self, "_draining", False)
                or owned_local_target(authority, source.chat_id, source=source) is None):
            return LOCAL_POINTER
        return result
