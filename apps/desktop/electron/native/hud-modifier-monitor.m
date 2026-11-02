// Passive, opt-in Cmd+Option tap. No character decoding, input log, or event IPC.
#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CoreGraphics.h>
#import <IOKit/hidsystem/IOLLEvent.h>
#include "hud-modifier-gesture.h"

// Transient physical state only, discarded on release/exit (never serialized).
typedef struct {
  HudModifierGesture gesture;
  bool keys[128];
  unsigned heldKeys;
  uint32_t buttons;
} HudMacState;

static bool TargetKey(CGKeyCode key) { return key == 54 || key == 55 || key == 58 || key == 61; }

static unsigned Modifiers(CGEventFlags flags) {
  CGEventFlags extra = kCGEventFlagMaskAlphaShift | kCGEventFlagMaskShift
    | kCGEventFlagMaskControl | kCGEventFlagMaskSecondaryFn | kCGEventFlagMaskHelp;
  return ((flags & kCGEventFlagMaskCommand) ? 1u : 0u)
    | ((flags & kCGEventFlagMaskAlternate) ? 2u : 0u) | ((flags & extra) ? 4u : 0u);
}

static bool HudMacUpdate(HudMacState *s, CGEventType type, CGKeyCode key,
                        CGEventFlags flags, bool repeat, unsigned button, uint64_t ms) {
  bool interrupted = repeat;
  unsigned modifiers = Modifiers(flags);
  switch (type) {
    case kCGEventFlagsChanged:
      interrupted |= !TargetKey(key) || modifiers == s->gesture.modifiers;
      break;
    case kCGEventKeyDown:
    case kCGEventKeyUp:
      interrupted = true;
      if (key < 128 && !TargetKey(key)) {
        bool down = type == kCGEventKeyDown;
        if (down != s->keys[key]) {
          s->heldKeys = down ? s->heldKeys + 1 : s->heldKeys - 1;
          s->keys[key] = down;
        }
      }
      break;
    case kCGEventLeftMouseDown:
    case kCGEventRightMouseDown:
    case kCGEventOtherMouseDown:
      if (button < 32) s->buttons |= 1u << button;
      interrupted = true;
      break;
    case kCGEventLeftMouseUp:
    case kCGEventRightMouseUp:
    case kCGEventOtherMouseUp:
      if (button < 32) s->buttons &= ~(1u << button);
      interrupted = true;
      break;
    case kCGEventMouseMoved:
      if (!s->buttons) return false;
      interrupted = true;
      break;
    default: // Wheel and drag always cancel, even if their button press was missed.
      interrupted = true;
      break;
  }
  return HudModifierUpdate(&s->gesture, modifiers, s->heldKeys || s->buttons, interrupted, ms);
}

#ifndef HUD_MODIFIER_MONITOR_TEST
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void Emit(const char *line) {
  size_t size = strlen(line);
  // A fixed, atomic, nonblocking message: a stalled/closed parent ends monitoring.
  if (write(STDOUT_FILENO, line, size) != (ssize_t)size) _exit(0);
}
static void Error(void) {
  Emit(CGPreflightListenEventAccess()
    ? "{\"type\":\"error\",\"code\":\"unavailable\"}\n"
    : "{\"type\":\"error\",\"code\":\"permission-required\"}\n");
}
static CGEventRef Observe(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *info) {
  (void)proxy;
  if (type == kCGEventTapDisabledByTimeout || type == kCGEventTapDisabledByUserInput) {
    // Never re-enable with stale state after lost events or permission revocation.
    Error();
    CFRunLoopStop(CFRunLoopGetMain());
    return event;
  }
  if (HudMacUpdate(info, type, (CGKeyCode)CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode),
      CGEventGetFlags(event), CGEventGetIntegerValueField(event, kCGKeyboardEventAutorepeat) != 0,
      (unsigned)CGEventGetIntegerValueField(event, kCGMouseEventButtonNumber), CGEventGetTimestamp(event) / 1000000)) {
    Emit("{\"type\":\"summon\"}\n");
  }
  return event; // The tap is listen-only: nothing is consumed or changed.
}

int main(int argc, const char *argv[]) { @autoreleasepool {
  signal(SIGPIPE, SIG_IGN);
  int flags = fcntl(STDOUT_FILENO, F_GETFL);
  if (flags < 0 || fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK) < 0) return 3;
  bool check = argc == 2 && strcmp(argv[1], "--check") == 0;
  bool request = argc == 2 && strcmp(argv[1], "--request-permission") == 0;
  if (argc > 1 && !check && !request) {
    Emit("{\"type\":\"error\",\"code\":\"unavailable\"}\n"); return 64;
  }
  bool allowed = CGPreflightListenEventAccess();
  if (!allowed && request) allowed = CGRequestListenEventAccess();
  if (!allowed) { Emit("{\"type\":\"error\",\"code\":\"permission-required\"}\n"); return 2; }
  HudMacState state = {0};
  CGEventMask mask = CGEventMaskBit(kCGEventFlagsChanged) | CGEventMaskBit(kCGEventKeyDown)
    | CGEventMaskBit(kCGEventKeyUp) | CGEventMaskBit(kCGEventLeftMouseDown) | CGEventMaskBit(kCGEventLeftMouseUp)
    | CGEventMaskBit(kCGEventRightMouseDown) | CGEventMaskBit(kCGEventRightMouseUp)
    | CGEventMaskBit(kCGEventOtherMouseDown) | CGEventMaskBit(kCGEventOtherMouseUp)
    | CGEventMaskBit(kCGEventScrollWheel) | CGEventMaskBit(kCGEventLeftMouseDragged)
    | CGEventMaskBit(kCGEventRightMouseDragged) | CGEventMaskBit(kCGEventOtherMouseDragged);
  CFMachPortRef tap = CGEventTapCreate(kCGSessionEventTap, kCGHeadInsertEventTap,
    kCGEventTapOptionListenOnly, mask, Observe, &state);
  if (!tap) { Error(); return 3; }
  CFRunLoopSourceRef source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0);
  if (!source) { CFMachPortInvalidate(tap); CFRelease(tap); Error(); return 3; }
  CFRunLoopAddSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes);
  // Seed once, after installing the tap. No polling or history of typed input.
  for (CGKeyCode key = 0; key < 128; key++) {
    // Flags-only keys (54..63) are represented by the modifier snapshot, not
    // heldKeys: they never produce the KeyUp that clears an ordinary key.
    if ((key < 54 || key > 63) && CGEventSourceKeyState(kCGEventSourceStateCombinedSessionState, key)) {
      state.keys[key] = true; state.heldKeys++;
    }
  }
  for (unsigned button = 0; button < 32; button++) {
    if (CGEventSourceButtonState(kCGEventSourceStateCombinedSessionState, button)) state.buttons |= 1u << button;
  }
  unsigned initial = Modifiers(CGEventSourceFlagsState(kCGEventSourceStateCombinedSessionState));
  HudModifierUpdate(&state.gesture, initial, state.heldKeys || state.buttons, initial != 0, 0);

  signal(SIGTERM, SIG_IGN);
  signal(SIGINT, SIG_IGN);
  dispatch_source_t term = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGTERM, 0, dispatch_get_main_queue());
  dispatch_source_t interrupt = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGINT, 0, dispatch_get_main_queue());
  dispatch_source_t input = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, STDIN_FILENO, 0, dispatch_get_main_queue());
  dispatch_block_t stop = ^{ CFRunLoopStop(CFRunLoopGetMain()); };
  dispatch_source_set_event_handler(term, stop);
  dispatch_source_set_event_handler(interrupt, stop);
  dispatch_source_set_event_handler(input, ^{
    char byte;
    if (read(STDIN_FILENO, &byte, 1) <= 0) CFRunLoopStop(CFRunLoopGetMain());
  });
  dispatch_resume(term); dispatch_resume(interrupt); dispatch_resume(input);
  CGEventTapEnable(tap, true);
  Emit("{\"type\":\"ready\"}\n");
  if (!check) CFRunLoopRun();
  dispatch_source_cancel(input); dispatch_source_cancel(interrupt); dispatch_source_cancel(term);
  CGEventTapEnable(tap, false); CFMachPortInvalidate(tap);
  CFRunLoopRemoveSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes);
  CFRelease(source); CFRelease(tap);
  return 0;
} }
#endif
