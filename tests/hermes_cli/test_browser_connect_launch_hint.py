"""Chrome's launch-failure hint names the stderr line that explains the exit.

``ChromeDebugLaunch.hint`` quoted the last line of Chrome's stderr tail, and Chrome's last line is often
noise: after the profile singleton refused the launch, the zygote logged "write: Broken pipe (32)". The
samples are the stderr of two real failed launches of Google Chrome 156 on Linux, as
``_read_stderr_tail`` returns them (paths and the other host's name shortened).
"""

from __future__ import annotations

import pytest

from hermes_cli.browser_connect import ChromeDebugLaunch, LaunchAttempt

# A HERMES_HOME long enough that Chrome's singleton socket path passed the AF_UNIX limit (exit -5).
SOCKET_PATH_TOO_LONG = "\n".join((
    "[1027/193440.991976:WARNING:chrome/app/chrome_main_linux.cc:84] Read channel stable from "
    "/opt/google/chrome/CHROME_VERSION_EXTRA",
    "[365:365:1027/193441.011830:FATAL:chrome/browser/process_singleton_posix.cc:315] Socket path too long: "
    "/srv/hermes/profiles-with-a-rather-long-parent-directory-name/a-long-profile-name/home/"
    "cache/scratch/com.google.Chrome.vs0Ijm/SingletonSocket.",
))
# The profile's singleton lock named another host, so Chrome refused the profile.
ZYGOTE_NOISE = (
    "[1:1:0100/000000.216990:ERROR:content/zygote/zygote_main_linux.cc:197] Failed sending zygote boot "
    "message: Broken pipe (32)",
    "[314:314:0100/000000.217052:ERROR:content/zygote/zygote_linux.cc:662] write: Broken pipe (32)",
)
PROFILE_IN_USE = "\n".join((
    "[1027/193941.186947:WARNING:chrome/app/chrome_main_linux.cc:84] Read channel stable from "
    "/opt/google/chrome/CHROME_VERSION_EXTRA",
    "[304:304:1027/193941.197742:ERROR:chrome/browser/process_singleton_posix.cc:367] The profile appears to be "
    "in use by another Google Chrome process (582) on another computer (another-computer).  Chrome has locked the "
    "profile so that it doesn't get corrupted.  If you are sure no other processes are using this profile, you "
    "can unlock the profile and relaunch Chrome.",
    "[304:304:1027/193941.200120:ERROR:chrome/browser/ui/dialogs/process_singleton_dialog_linux.cc:93] Cannot "
    "show profile-in-use dialog: no system dialog tool (zenity, kdialog, xmessage) was found. To recover, delete "
    "the files SingletonLock, SingletonCookie, SingletonSocket in your profile directory.",
    *ZYGOTE_NOISE,
))


def _hint(stderr_tail: str, returncode: int) -> str | None:
    attempt = LaunchAttempt(
        binary="/opt/google/chrome/chrome", state="exited", returncode=returncode, stderr_tail=stderr_tail)
    return ChromeDebugLaunch(attempts=[attempt]).hint


@pytest.mark.parametrize(("stderr_tail", "returncode", "explains"), [
    pytest.param(PROFILE_IN_USE, 21, "The profile appears to be in use by another Google Chrome process (582)",
                 id="profile-in-use"),
    pytest.param(SOCKET_PATH_TOO_LONG, -5, "FATAL:chrome/browser/process_singleton_posix.cc:315] Socket path too "
                 "long", id="socket-path-too-long"),
    # The cause still wins when Chrome's helpers log after it (the socket log, then the zygote lines).
    pytest.param("\n".join((SOCKET_PATH_TOO_LONG, *ZYGOTE_NOISE)), -5, "Socket path too long",
                 id="socket-path-too-long-before-noise"),
    # A FATAL line with no known cause in it (made up, in Chromium's log format) beats the noise too.
    pytest.param("\n".join((SOCKET_PATH_TOO_LONG.splitlines()[0],
                            "[7:7:1027/193441.0:FATAL:example/startup.cc:1] a failure no pattern names",
                            *ZYGOTE_NOISE)), -5, "FATAL:example/startup.cc:1] a failure no pattern names",
                 id="other-fatal-before-noise"),
])
def test_the_hint_quotes_the_line_that_explains_the_exit(stderr_tail, returncode, explains):
    hint = _hint(stderr_tail, returncode)

    assert hint.startswith("chrome exited before the debug port opened: ")
    assert explains in hint
    assert "Broken pipe" not in hint


def test_without_an_explaining_line_the_hint_quotes_the_last_line():
    hint = _hint("\n".join(("[1027/1.0:WARNING:chrome/app/chrome_main_linux.cc:84] Read channel stable",
                            "error while loading shared libraries: libnspr4.so")), 127)

    assert hint == "chrome exited before the debug port opened: error while loading shared libraries: libnspr4.so"
