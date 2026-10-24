"""QQ HTTP and business-error responses shared by connected and standalone sends."""


class QQAPIError(RuntimeError):
    """The QQ response explicitly reports an HTTP or business error."""

    def __init__(self, status_code: int, path: str, code, detail):
        self.status_code = status_code
        self.code = code
        # Keep HTTP-error text unchanged: existing retry policy inspects it.
        reason = f"code={code} {detail}" if status_code < 400 else detail
        super().__init__(f"QQ Bot API error [{status_code}] {path}: {reason}")

    @property
    def business_error(self) -> bool:
        return 200 <= self.status_code < 300 and self.code not in (None, 0, "0")


def raise_for_qq_error(status_code: int, path: str, data) -> None:
    """HTTP 2xx does not override a nonzero QQ business code."""
    code = data.get("code") if isinstance(data, dict) else None
    if status_code >= 400 or code not in (None, 0, "0"):
        detail = data.get("message", data) if isinstance(data, dict) else data
        raise QQAPIError(status_code, path, code, detail)
