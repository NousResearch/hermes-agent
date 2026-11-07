"""The real browser profile's saved passwords as a READ-ONLY vault login source (``browser:``).

Enabled exactly when ``browser.use_real_profile`` is on — the same consent that already copies the
user's real Chrome/Brave profile (cookies AND ``Login Data``) for the agent's browser and that
PR #150283 used to carry the real-profile cookies into the Desktop in-app browser. This backend
carries the saved PASSWORDS in the same way: the entries appear in the in-app Browser bar's
key-button picker alongside the Hermes vault / Bitwarden / 1Password sources and fill password-blind
on click.

There is no decryption code here, and no new consent key. The reads go over CDP against the
already-running real-profile browser (``tools.browser_real_profile_logins``):

- ``list_items`` → origin + username only, read from the plaintext columns of the profile's snapshot
  ``Login Data`` (never the encrypted password).
- ``resolve_password`` → the browser autofills the saved password on the credential's own origin and
  Hermes reads it back over CDP at fill time (the browser decrypts its OWN store). The value is
  resolved server-side only, never listed, never logged; the fill tool registers it with the
  redaction boundary like every other vault fill.

Read-only: no ``add``/``remove``. Per Teknium's ruling this adds NO restriction on the agent's own
browser — it only makes the saved logins visible and fillable.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)


class BrowserProfileLoginBackend(LoginBackend):
    name = "browser"
    display_name = "Browser profile"
    prefix = "browser:"
    needs_unlock = False  # the real browser already decrypts its own store; no master password.

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        # Name the detected browser so the picker shows e.g. "Browser profile — Chrome".
        self.display_name = f"Browser profile — {self._browser_label()}"

    # ── detection / snapshot location ─────────────────────────────────────────────────────────────

    def _browser_name(self) -> Optional[str]:
        from hermes_cli.browser_connect import detect_default_chromium
        return detect_default_chromium()

    def _browser_label(self) -> str:
        key = self._browser_name() or "Chromium"
        return {"chrome": "Chrome", "chromium": "Chromium", "brave": "Brave",
                "brave-origin": "Brave Origin", "microsoft-edge": "Edge"}.get(key, key.title())

    def _copy_dir(self) -> Optional[str]:
        browser = self._browser_name()
        if not browser:
            return None
        from hermes_cli.browser_connect import real_profile_copy_dir
        return real_profile_copy_dir(browser)

    def _ensure_snapshot(self) -> Optional[str]:
        """The snapshot copy dir with a populated ``Login Data``, preparing it first when needed.

        A freshly-enabled source on a profile that has had no prior browser session has no snapshot
        yet, so a bare read would list empty and the user could never select a login to initialise it.
        Prepare the consented snapshot through the SAME lifecycle a fill uses (``real_profile_cdp()``,
        which snapshots then launches/attaches the real-profile browser) before reading. The launch is
        idempotent and shared per profile, so listing reuses an already-running browser. Returns
        ``None`` when consent is off or the snapshot still cannot be produced (listing stays empty, but
        only because there is genuinely nothing to show)."""
        copy_dir = self._copy_dir()
        if copy_dir and self._login_data_present(copy_dir):
            return copy_dir
        # No usable snapshot yet: drive the real-profile lifecycle to create one. Failure (consent
        # off, unsupported browser, locked profile) is non-fatal here — fall back to whatever the copy
        # dir holds (usually nothing), so listing degrades to empty rather than raising.
        try:
            from tools.browser_session_real_profile import real_profile_cdp
            real_profile_cdp()
        except Exception as exc:  # health: allow BLE001 -- listing must never raise; a later fill surfaces the real error
            logger.debug("real-profile logins: snapshot preparation failed: %s", exc)
        copy_dir = self._copy_dir()
        return copy_dir if copy_dir and self._login_data_present(copy_dir) else None

    @staticmethod
    def _login_data_present(copy_dir: str) -> bool:
        from tools.browser_real_profile_logins import login_data_present
        return login_data_present(copy_dir)

    def _handle(self, nav_url: str, username: str) -> str:
        # Opaque handle: prefix + saved-login URL + "\x1f" + username. The URL (path preserved) and
        # username are metadata (the agent already sees them in list output), so no secret is encoded
        # here. The full URL is carried, not just the origin, so a login saved at
        # ``https://example.com/account/login`` is resolved on that page, not the bare root.
        return f"{self.prefix}{nav_url}\x1f{username}"

    def _split(self, handle: str) -> tuple[str, str]:
        body = handle[len(self.prefix):]
        nav_url, _, username = body.partition("\x1f")
        return nav_url, username

    # ── LoginBackend contract ──────────────────────────────────────────────────────────────────────

    def list_items(self) -> list[VaultItemMeta]:
        copy_dir = self._ensure_snapshot()
        if not copy_dir:
            return []
        from tools.browser_real_profile_logins import read_login_metadata
        from agent.vault_store import VaultError
        out: list[VaultItemMeta] = []
        for row in read_login_metadata(copy_dir):
            nav_url = row["origin_url"].strip()
            try:
                origin = normalize_origin(nav_url)
            except VaultError:
                continue
            if not origin or not origin.startswith(("http://", "https://")):
                continue  # fill targets are web pages; app/extension origins never widen the set
            username = row["username"].strip() or None
            out.append(VaultItemMeta(
                # The handle carries the full saved URL (path preserved) so resolution opens the exact
                # sign-in page, while ``origin``/``allowed_origins`` stay the normalized origin the
                # fill tool's exact-origin guard checks against.
                id=self._handle(nav_url, username or ""), kind="login",
                label=f"{origin.split('://', 1)[-1]}" + (f" ({username})" if username else ""),
                origin=origin, created_at="",
                identifier_type="username" if username else None, identifier=username,
                allowed_origins=(origin,)))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        return next((m for m in self.list_items() if m.id == handle), None)

    def resolve_password(self, handle: str) -> str:
        nav_url, username = self._split(handle)
        from agent.vault_store import VaultError
        try:
            origin = normalize_origin(nav_url)
        except VaultError:
            origin = nav_url
        from tools.browser_session_real_profile import real_profile_cdp
        cdp_url, err = real_profile_cdp()
        if cdp_url is None:
            raise RuntimeError(err or "real-profile browsing is off; cannot resolve the saved password")
        from tools.browser_real_profile_logins import resolve_login_password
        # nav_url opens the exact saved sign-in page (path preserved); origin is the origin the final
        # page must still be on after any redirect before the autofilled value is accepted.
        password = resolve_login_password(cdp_url, nav_url, origin, username)
        if not password:
            raise RuntimeError(
                "The browser did not autofill a saved password for this login. It may have been "
                "removed, need a chooser (several accounts on this site), redirect off its origin, or "
                "be unavailable in this session. Save it to the Hermes vault, or sign in in the browser.")
        return password
