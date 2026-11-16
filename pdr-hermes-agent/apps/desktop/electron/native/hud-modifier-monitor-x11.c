// XI2 raw events are passive; no grabs, remapping, characters, or event logging.
#include <X11/Xlib.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>
#include <X11/extensions/XInput2.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>
#include "hud-modifier-gesture.h"

static HudModifierGesture gesture;
static bool keys[256]; // Current physical state only, never emitted.
static unsigned char targets[256];
static bool buttons[256];
static volatile sig_atomic_t stopping;
static void Stop(int signalNumber) { (void)signalNumber; stopping = 1; }
static uint64_t Now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000 + (uint64_t)t.tv_nsec / 1000000;
}
static void Emit(const char *line) {
  size_t size = strlen(line);
  if (write(STDOUT_FILENO, line, size) != (ssize_t)size) _exit(0);
}
static int Unavailable(void) { Emit("{\"type\":\"error\",\"code\":\"unavailable\"}\n"); return 3; }
static int XFailure(Display *display, XErrorEvent *error) {
  (void)display; (void)error;
  Unavailable(); _exit(3);
}
static int XIOFailure(Display *display) { (void)display; Unavailable(); _exit(3); }

static void Update(bool interrupted) {
  unsigned modifiers = 0, first = 0, second = 0;
  bool held = false;
  for (unsigned key = 0; key < 256; key++) {
    if (!keys[key]) continue;
    if (targets[key] == 1) first++;
    else if (targets[key] == 2) second++;
    else held = true;
  }
  for (unsigned button = 0; button < 256; button++) held |= buttons[button];
  if (first) modifiers |= 1;
  if (second) modifiers |= 2;
  if (first > 1 || second > 1) modifiers |= 4;
  if (HudModifierUpdate(&gesture, modifiers, held, interrupted, Now())) Emit("{\"type\":\"summon\"}\n");
}

static void Snapshot(Display *display) {
  memset(keys, 0, sizeof(keys));
  memset(buttons, 0, sizeof(buttons));
  memset(targets, 0, sizeof(targets));
  KeySym symbols[] = { XK_Control_L, XK_Control_R, XK_Alt_L, XK_Alt_R };
  for (unsigned i = 0; i < 4; i++) {
    KeyCode code = XKeysymToKeycode(display, symbols[i]);
    // Do not turn AltGr / level-switch keys into a Ctrl+Alt tap.
    if (code && XkbKeycodeToKeysym(display, code, 0, 0) == symbols[i]) targets[code] = i < 2 ? 1 : 2;
  }
  char map[32];
  XQueryKeymap(display, map);
  for (unsigned key = 0; key < 256; key++) keys[key] = ((unsigned char)map[key / 8] & (1u << (key % 8))) != 0;
  int count = 0;
  XIDeviceInfo *devices = XIQueryDevice(display, XIAllMasterDevices, &count);
  if (!devices) { Unavailable(); _exit(3); }
  for (int device = 0; device < count; device++) {
    for (int c = 0; c < devices[device].num_classes; c++) {
      if (devices[device].classes[c]->type != XIButtonClass) continue;
      XIButtonClassInfo *b = (XIButtonClassInfo *)devices[device].classes[c];
      for (int button = 0; button < 256 && button < b->state.mask_len * 8; button++) {
        buttons[button] |= XIMaskIsSet(b->state.mask, button) != 0;
      }
    }
  }
  XIFreeDeviceInfo(devices);
  gesture = (HudModifierGesture){0};
  Update(true); // An already-held chord cannot authorize a summon.
}

int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN); signal(SIGINT, Stop); signal(SIGTERM, Stop);
  int outFlags = fcntl(STDOUT_FILENO, F_GETFL);
  if (outFlags < 0 || fcntl(STDOUT_FILENO, F_SETFL, outFlags | O_NONBLOCK) < 0) return 3;
  bool check = argc == 2 && strcmp(argv[1], "--check") == 0;
  if (argc > 1 && !check && !(argc == 2 && strcmp(argv[1], "--request-permission") == 0)) return Unavailable();
  const char *wayland = getenv("WAYLAND_DISPLAY"), *session = getenv("XDG_SESSION_TYPE");
  if ((wayland && *wayland) || (session && strcasecmp(session, "wayland") == 0)) return Unavailable();
  Display *display = XOpenDisplay(NULL);
  if (!display) return Unavailable();
  XSetErrorHandler(XFailure); XSetIOErrorHandler(XIOFailure);
  int opcode, event, error;
  if (XQueryExtension(display, "XWAYLAND", &opcode, &event, &error)
      || !XQueryExtension(display, "XInputExtension", &opcode, &event, &error)) {
    XCloseDisplay(display); return Unavailable();
  }
  int major = 2, minor = 2;
  if (XIQueryVersion(display, &major, &minor) != Success || major < 2 || (major == 2 && minor < 2)) {
    XCloseDisplay(display); return Unavailable();
  }
  Bool detectable = False;
  XkbSetDetectableAutoRepeat(display, True, &detectable);
  if (!detectable) { XCloseDisplay(display); return Unavailable(); }
  unsigned char rawMask[XIMaskLen(XI_LASTEVENT)] = {0};
  XISetMask(rawMask, XI_RawKeyPress); XISetMask(rawMask, XI_RawKeyRelease);
  XISetMask(rawMask, XI_RawButtonPress); XISetMask(rawMask, XI_RawButtonRelease);
  XISetMask(rawMask, XI_RawMotion);
  unsigned char devicesMask[XIMaskLen(XI_LASTEVENT)] = {0};
  XISetMask(devicesMask, XI_HierarchyChanged);
  XIEventMask masks[] = {
    { XIAllMasterDevices, sizeof(rawMask), rawMask },
    { XIAllDevices, sizeof(devicesMask), devicesMask }
  };
  XISelectEvents(display, DefaultRootWindow(display), masks, 2);
  XSync(display, False);
  Snapshot(display);
  Emit("{\"type\":\"ready\"}\n");
  struct pollfd fds[] = { { ConnectionNumber(display), POLLIN, 0 }, { STDIN_FILENO, POLLIN, 0 } };
  while (!check && !stopping) {
    while (XPending(display) && !stopping) {
      XEvent input;
      XNextEvent(display, &input);
      if (input.type == MappingNotify) { XRefreshKeyboardMapping(&input.xmapping); Snapshot(display); continue; }
      if (input.type != GenericEvent || input.xcookie.extension != opcode || !XGetEventData(display, &input.xcookie)) continue;
      if (input.xcookie.evtype == XI_HierarchyChanged) {
        Snapshot(display);
      } else {
        XIRawEvent *raw = input.xcookie.data;
        int code = raw->detail;
        bool interrupted = false;
        if (raw->evtype == XI_RawKeyPress || raw->evtype == XI_RawKeyRelease) {
          if (code < 0 || code >= 256) interrupted = true;
          else {
            bool down = raw->evtype == XI_RawKeyPress;
            interrupted = !targets[code] || keys[code] == down || (raw->flags & XIKeyRepeat);
            keys[code] = down;
          }
        } else if (raw->evtype == XI_RawButtonPress || raw->evtype == XI_RawButtonRelease) {
          // X11 wheel events are buttons as well.
          if (code >= 0 && code < 256) buttons[code] = raw->evtype == XI_RawButtonPress;
          interrupted = true;
        } else if (raw->evtype == XI_RawMotion) {
          for (unsigned b = 0; b < 256; b++) interrupted |= buttons[b];
        }
        Update(interrupted);
      }
      XFreeEventData(display, &input.xcookie);
    }
    if (stopping) break;
    int result = poll(fds, 2, -1);
    if (result < 0) { if (errno == EINTR) continue; Unavailable(); break; }
    if (fds[1].revents & (POLLHUP | POLLERR | POLLNVAL)) break;
    if (fds[1].revents & POLLIN) { char byte; if (read(STDIN_FILENO, &byte, 1) <= 0) break; }
    if (fds[0].revents & (POLLHUP | POLLERR | POLLNVAL)) { Unavailable(); break; }
  }
  XCloseDisplay(display);
  return 0;
}
