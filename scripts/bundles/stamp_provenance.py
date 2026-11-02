"""Native artifact stamp custody: desktop channel requests versus APT-owned runtimes."""
from hermes_cli.update_channel import is_channel_tag, is_commit_tag, release_version


def _channel_stamp_matches(stamp: dict, tag: str, commit: str, request: dict) -> None:
    if (stamp.get("source") != "channel-build" or stamp.get("channelBuild") != request
            or stamp.get("commit") != request["commit"] or commit != request["commit"]
            or stamp.get("tag") != request["releaseTag"] or tag != request["releaseTag"]):
        raise ValueError("Built package provenance does not match the channel request")


def stamp_matches(stamp: dict, tag: str, commit: str, *, channel_request: dict | None = None,
                  termux: bool = False) -> None:
    if termux:
        owner = tuple(stamp.get(key) for key in ("payload", "distribution", "updateMechanism"))
        if owner != ("runtime", "apt-termux", "external") or channel_request is not None:
            raise ValueError("Termux package update ownership must be an external apt-termux runtime")
    elif channel_request is not None:
        _channel_stamp_matches(stamp, tag, commit, channel_request)
        return
    elif is_channel_tag(tag):
        raise ValueError("A desktop channel package requires its admitted channel request")
    version = release_version(tag)
    if (version is None or stamp.get("commit") != commit or stamp.get("tag") != tag
            or stamp.get("baseVersion") != version):
        raise ValueError("Built package provenance does not match the release")
    if (stamp.get("source") == "commit-build") != is_commit_tag(tag):
        raise ValueError("Built package update ownership does not match the release kind")
