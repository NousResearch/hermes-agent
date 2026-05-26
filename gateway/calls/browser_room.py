from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from urllib.parse import quote, urlsplit

from .models import CallError

_TAILSCALE_CGNAT = ip_network("100.64.0.0/10")


@dataclass(frozen=True)
class BrowserRoomConfig:
    base_url: str = ""
    public_exposure_enabled: bool = False


def is_tailnet_url(url: str) -> bool:
    parsed = urlsplit(str(url or "").strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if not parsed.scheme or parsed.scheme not in {"https", "wss"} or not host:
        return False
    if host.endswith(".ts.net"):
        return True
    try:
        return ip_address(host) in _TAILSCALE_CGNAT
    except ValueError:
        return False


class BrowserRoomProvider:
    def __init__(self, config: BrowserRoomConfig):
        self.config = config

    def create_room_url(self, call_id: str, token: str) -> str:
        base_url = str(self.config.base_url or "").strip().rstrip("/")
        if not base_url:
            raise CallError(
                "call_public_url_missing",
                "Call setup failed: browser call URL is not configured.",
            )
        if not self.config.public_exposure_enabled and not is_tailnet_url(base_url):
            raise CallError(
                "call_public_exposure_disabled",
                "Call setup failed: browser call URL must be Tailnet-only.",
            )
        return f"{base_url}/{quote(call_id, safe='')}?token={quote(token, safe='')}"
