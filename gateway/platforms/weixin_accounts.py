"""Retire superseded Weixin identities after a successful profile-local rebind."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def clear_stale_weixin_accounts(home: str, current_account_id: str, user_id: str) -> None:
    from gateway.platforms.weixin import list_weixin_accounts
    from gateway.platforms.weixin_quotes import WeixinQuoteStore

    if not user_id:
        return
    root = Path(home) / "weixin" / "accounts"
    for account in list_weixin_accounts(home):
        account_id = account["account_id"]
        if account_id == current_account_id or account["user_id"].strip() != user_id:
            continue
        # Only a confirmed replacement for the same user invalidates the old bot identity.
        WeixinQuoteStore(home, account_id).remove_account()
        try:
            for suffix in (".sync.json", ".context-tokens.json", ".json"):
                (root / f"{account_id}{suffix}").unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Weixin superseded account cleanup failed: %s", exc)
