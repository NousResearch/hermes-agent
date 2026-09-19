"""Journal-backed retain durability: a payload is written to disk BEFORE it is
handed to the writer thread, so a crash between enqueue and the network call
cannot silently drop a turn. Drives the same writer queue/thread the provider
already has — this module only decides what gets journaled and replayed.
"""
import hashlib
import json
import logging

from hermes_constants import get_hermes_home

from .outbox import Outbox

logger = logging.getLogger(__name__)
UNCONFIRMED = ("Hindsight memory unavailable: prior retention is not confirmed; "
               "pending work requires retry or owner review.")


class RetainReliability:
    def _outbox(self) -> Outbox:
        """Lazily open the journal, bound to this provider's (home, mode, api_url,
        bank, key) partition. Reused across calls; refuses to be reused if that
        identity ever changes under the same instance (would mix two tenants'
        journals)."""
        home = get_hermes_home().resolve()
        partition = hashlib.sha256(json.dumps([
            self._mode, self._api_url.rstrip("/"), self._bank_id, self._api_key or "",
        ]).encode()).hexdigest()
        binding = (home, partition)
        if self._journal is None:
            self._journal = Outbox(home, partition)
            self._journal_binding = binding
        elif binding != self._journal_binding:
            raise RuntimeError("Hindsight provider used outside its owning profile/endpoint/bank")
        return self._journal

    def _recover_retains(self) -> None:
        """Called from initialize(): replay work a prior process left durable but
        unconfirmed (queued, or interrupted mid-send)."""
        for identity in self._outbox().recover():
            self._enqueue_retain(lambda identity=identity: self._process_retain(identity))

    def _process_retain(self, identity: str) -> None:
        """Writer-thread job: claim the journaled row, send it via the existing
        legacy retain call, and record the outcome. Never resubmits a row whose
        send may already have reached the server — see Outbox.finish/requeue."""
        journal = self._outbox()
        payload = journal.claim(identity)
        if payload is None:
            return  # already claimed/finished (e.g. a duplicate recover() call)
        try:
            response = self._retain_batch(
                payload["item"], bank_id=payload["bank_id"],
                document_id=payload.get("document_id"), retain_async=payload.get("retain_async"),
            )
        except Exception:
            journal.finish(identity, "quarantined")
            logger.warning(UNCONFIRMED)
            return
        if payload.get("retain_async") and payload.get("track_ops", True):
            self._track_retain_ops(response, payload["bank_id"])
        journal.finish(identity, "done")

    def _warn_if_unconfirmed(self) -> None:
        """Shutdown-time visibility: a journal that still has queued/sending/
        quarantined rows means memory the user believes was saved is not
        confirmed yet. Reads the already-open journal directly (never calls
        _outbox(), which would re-validate the profile/endpoint/bank binding —
        shutdown may run after a caller mutated mode/url/bank on a live instance)."""
        journal = self._journal
        if journal is not None and journal.pending():
            logger.warning(UNCONFIRMED)
