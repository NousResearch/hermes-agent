"""Inert peer/local transport fixtures for canonical Retry controls.

Extracted unchanged from the accepted historical member-progress witness.
Source: 50b52142457bf4a50af229aa3e92de60816d5dc2,
 tests/tui_gateway/test_hosted_room_peer_backoff_progress.py.
Historical assertion provenance: David Dudok de Wit, 691bb08a5cd310fffc5f3d01653dc93f394fc080.
"""
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


class LocalRPC:
    def __init__(self):
        self.calls = []

    def resolve_exact(self, **kwargs):
        return {"session_id": "local-session"}

    def resume(self, **kwargs):
        return {"session_id": "local-session"}

    def begin_attachment_staging(self, **kwargs):
        pass

    def stage_attachment(self, **kwargs):
        self.calls.append(("stage", kwargs))
        return {"ref_text": "@file:attachments/notes.txt"}

    def commit_attachment_staging(self, **kwargs):
        pass

    def rollback_attachment_staging(self, **kwargs):
        pass

    def submit(self, **kwargs):
        self.calls.append(("submit", kwargs))
        kwargs["on_terminal"]({"status": "settled", "text": "reply from local"})
        return {"accepted": True}


class Peer:
    def __init__(self, mode):
        self.mode, self.staged, self.dispatches = mode, [], []
        self.on_stage = None

    def prepare(self, **kwargs):
        return {"session_id": "peer-session"}

    def stage_attachments(self, **kwargs):
        self.staged.append(kwargs)
        if self.on_stage:
            self.on_stage()
        if self.mode == "permission":
            raise PeerRunsHTTPError("permission denied", status_code=403,
                error_code="invalid_room_grant", not_admitted=True, retryable=False)
        if self.mode in {"unavailable", "ambiguous-upload"}:
            raise PeerRunsHTTPError("upload unavailable", retryable=True,
                not_admitted=self.mode == "unavailable", ambiguous=self.mode == "ambiguous-upload")
        return {"complete": True, "count": len(kwargs["attachments"])}

    def dispatch(self, **kwargs):
        self.dispatches.append(kwargs)
        if self.mode == "ambiguous-dispatch":
            raise PeerRunsHTTPError("admission response lost", retryable=True, ambiguous=True)
        return {"status": "settled", "text": "reply from peer"}

    def discard_attachments(self, **kwargs):
        return {"discarded": True}

    def history(self, **kwargs):
        return []

    def status(self, **kwargs):
        return {"active": False}
