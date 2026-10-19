"""The slow steps of a plugin install, as ids. A leaf module (stdlib only) so the gateway wire contract
(``tui_gateway/contracts/connectors_operation.py``) can type them without importing the installer."""

from enum import StrEnum


class InstallPhase(StrEnum):
    """The slow steps of an install, as ids; the desktop catalog card words them in its own language."""

    downloading = "downloading"
    python_packages = "python_packages"
    loading_tools = "loading_tools"
