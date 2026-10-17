"""Headers Hermes sends on its model calls, so an endpoint — or a gateway in front of one — can tell
which app a call came from.

A helper path that builds its own SDK client must send them too: with the SDK's default User-Agent
(``OpenAI/Python x.y``) the call reads downstream as a client of its own — magpie, for one, counts
usage by that header — instead of as Hermes'.
"""
from hermes_cli.version_info import get_version_info


def user_agent_headers() -> dict[str, str]:
    """The User-Agent every Hermes model call carries: ``HermesAgent/<base version>``."""
    return {"User-Agent": f"HermesAgent/{get_version_info().base_version}"}
