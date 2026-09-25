"""Exercise the actual XI2 helper on a PRIVATE Xvfb server, never the user's display.

Usage: python3 hud-modifier-x11.test.py /path/to/hud-modifier-monitor
Requires Xvfb and libXtst. No Desktop launch, global host input, or permission prompt.
"""
import json
import ctypes
import os
import select
import subprocess
import sys
import time


def message(child, timeout=5):
    assert select.select([child.stdout], [], [], timeout)[0], "helper timed out"
    line = child.stdout.readline()
    assert line, "helper exited unexpectedly"
    return json.loads(line)


def main(binary):
    read_fd, write_fd = os.pipe()
    server = subprocess.Popen(
        ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "640x480x24", "-nolisten", "tcp", "-ac"],
        pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    os.close(write_fd)
    child = None
    try:
        assert select.select([read_fd], [], [], 10)[0], "Xvfb did not start"
        display = os.read(read_fd, 32).decode().strip()
        assert display.isdigit(), "Xvfb did not supply a private display"
        env = {**os.environ, "DISPLAY": ":" + display, "XDG_SESSION_TYPE": "x11"}
        env.pop("WAYLAND_DISPLAY", None)
        # Keep one injector connection alive. Starting a new xdotool process
        # enables/disables XTEST devices, correctly invalidating the helper's
        # state via XI_HierarchyChanged in the middle of a gesture.
        x11 = ctypes.CDLL("libX11.so.6")
        xtst = ctypes.CDLL("libXtst.so.6")
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
        x11.XStringToKeysym.restype = ctypes.c_ulong
        x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        xtst.XTestFakeRelativeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
        injector = x11.XOpenDisplay(env["DISPLAY"].encode())
        assert injector
        def keys(*args):
            pending = iter(args)
            for command in pending:
                value = next(pending)
                if command == "sleep":
                    time.sleep(float(value))
                elif command in ("keydown", "keyup", "key"):
                    code = x11.XKeysymToKeycode(injector, x11.XStringToKeysym(value.encode()))
                    assert code
                    xtst.XTestFakeKeyEvent(injector, code, command != "keyup", 0)
                    if command == "key": xtst.XTestFakeKeyEvent(injector, code, False, 0)
                elif command in ("mousedown", "mouseup", "click"):
                    xtst.XTestFakeButtonEvent(injector, int(value), command != "mouseup", 0)
                    if command == "click": xtst.XTestFakeButtonEvent(injector, int(value), False, 0)
                elif command == "move":
                    xtst.XTestFakeRelativeMotionEvent(injector, int(value), int(next(pending)), 0)
                else:
                    raise AssertionError(command)
                x11.XSync(injector, False)
                time.sleep(0.015)
        keys("key", "Shift_L", "click", "1")
        child = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        assert child.stdin is not None and child.stdout is not None and child.stderr is not None
        assert message(child) == {"type": "ready"}
        def press():
            keys("keydown", "Control_L", "keydown", "Alt_L")
        def release():
            keys("keyup", "Control_L", "keyup", "Alt_L")
        press()
        release()
        assert message(child) == {"type": "summon"}
        for interference in [
            ("key", "x"), ("key", "Shift_L"), ("click", "1"), ("click", "4"),
            ("mousedown", "1", "move", "10", "10", "mouseup", "1"),
        ]:
            press()
            keys(*interference)
            release()
            assert not select.select([child.stdout], [], [], 2)[0], ("unexpected summon", interference)
        # Time is measured from first press until BOTH releases, not chord formation.
        press()
        keys("sleep", "0.6")
        release()
        assert not select.select([child.stdout], [], [], 2)[0], "long hold summoned"
        # EOF retires the actual helper promptly.
        child.stdin.close()
        assert child.wait(timeout=5) == 0
        assert child.stderr.read() == b""
        # Even DISPLAY pointing at X11 must not claim global access on Wayland.
        denied = subprocess.run([binary, "--check"], env={**env, "WAYLAND_DISPLAY": "wayland-test"}, capture_output=True, timeout=5)
        assert json.loads(denied.stdout) == {"type": "error", "code": "unavailable"}
        print("isolated XI2: summon, key/modifier/mouse/wheel/drag cancellation, duration, EOF, and Wayland checks passed")
    finally:
        os.close(read_fd)
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        server.terminate()
        server.wait(timeout=5)


if __name__ == "__main__":
    main(sys.argv[1])
