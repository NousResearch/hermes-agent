// Passive modifier gesture detection. Never reads characters, activates an app,
// captures pixels, or writes files. stdout is a bounded JSON-lines protocol.
#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CoreGraphics.h>
#import <IOKit/hidsystem/IOLLEvent.h>
#include <stdbool.h>
#include <stdio.h>

// Only physical key state is retained, not characters or an event history.
typedef struct {
  bool left;
  bool right;
  bool blocked;
  bool keys[128];
  unsigned heldKeys;
} CommandGesture;

static bool CommandGestureUpdate(CommandGesture *state, CGEventType type,
                                  CGKeyCode key, CGEventFlags flags, bool repeat) {
  bool left = (flags & NX_DEVICELCMDKEYMASK) != 0;
  bool right = (flags & NX_DEVICERCMDKEYMASK) != 0;
  bool any = left || right || (flags & kCGEventFlagMaskCommand);
  bool wasBoth = state->left && state->right;
  bool commandEvent = type == kCGEventFlagsChanged && (key == 54 || key == 55);
  CGEventFlags forbidden = kCGEventFlagMaskAlphaShift | kCGEventFlagMaskShift
    | kCGEventFlagMaskControl | kCGEventFlagMaskAlternate | kCGEventFlagMaskSecondaryFn
    | kCGEventFlagMaskHelp | kCGEventFlagMaskNumericPad;

  if (type == kCGEventKeyDown) {
    state->blocked = true;
    if (key < 128 && !state->keys[key]) { state->keys[key] = true; state->heldKeys++; }
  } else if (type == kCGEventKeyUp) {
    state->blocked = true; // also reject keys already held when monitoring started
    if (key < 128 && state->keys[key]) {
      state->keys[key] = false;
      state->heldKeys--;
    }
  }
  if ((flags & forbidden) || state->heldKeys || repeat
      || (any && !(left || right))) state->blocked = true;

  // Capture on the clean second physical press, while both keys are still down.
  // Subsequent letters cannot retroactively cancel an already authorized capture.
  bool capture = false;
  if (commandEvent && left && right && !wasBoth) {
    // Both physical transitions must be seen. An initial or ambiguous held chord
    // is not evidence of a gesture and is ignored until both keys are released.
    bool secondPress = (key == 54 && state->left && !state->right)
      || (key == 55 && state->right && !state->left);
    capture = secondPress && !state->blocked;
    state->blocked = true; // no repeat or partial-release re-arming
  }
  state->left = left;
  state->right = right;
  if (!any) state->blocked = state->heldKeys > 0 || (flags & forbidden) != 0;
  return capture;
}

static NSDictionary *CommandCaptureWindow(NSArray *windows, pid_t frontmostPID) {
  if (frontmostPID <= 0) return nil;
  // CGWindowListCopyWindowInfo returns front-to-back order. Restrict to the
  // foreground app, not an overlay or a background window behind the desktop.
  for (NSDictionary *window in windows) {
    NSNumber *pid = window[(id)kCGWindowOwnerPID];
    NSNumber *layer = window[(id)kCGWindowLayer];
    NSNumber *wid = window[(id)kCGWindowNumber];
    if (!pid || pid.intValue != frontmostPID || !layer || layer.intValue != 0
        || ![window[(id)kCGWindowIsOnscreen] boolValue]
        || [window[(id)kCGWindowAlpha] doubleValue] <= 0
        || !wid || wid.longLongValue <= 0 || wid.unsignedLongLongValue > UINT32_MAX) continue;
    NSDictionary *bounds = window[(id)kCGWindowBounds];
    CGRect rect;
    if (![bounds isKindOfClass:NSDictionary.class]
        || !CGRectMakeWithDictionaryRepresentation((__bridge CFDictionaryRef)bounds, &rect)
        || !isfinite(rect.size.width) || !isfinite(rect.size.height)
        || rect.size.width <= 0 || rect.size.height <= 0) continue;
    return @{ @"type": @"capture", @"windowId": wid,
      @"width": @(rect.size.width), @"height": @(rect.size.height) };
  }
  return nil;
}

#ifndef COMMAND_SCREENSHOT_MONITOR_TEST
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

static void EmitJSON(NSDictionary *message) {
  NSMutableData *line = [[NSJSONSerialization dataWithJSONObject:message options:0 error:nil] mutableCopy];
  [line appendBytes:"\n" length:1];
  // Never block the input callback on a stalled parent. Every protocol message
  // fits a single atomic pipe write; a closed/full pipe ends this helper.
  if (!line || line.length > 512 || write(STDOUT_FILENO, line.bytes, line.length) != (ssize_t)line.length) _exit(0);
}

static void EmitError(NSString *code) {
  EmitJSON(@{ @"type": @"error", @"code": code });
}

static CGEventRef ObserveEvent(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *info) {
  (void)proxy;
  @autoreleasepool {
    if (type == kCGEventTapDisabledByTimeout || type == kCGEventTapDisabledByUserInput) {
      // Missed input makes physical state unknowable. Fail closed, with no
      // automatic restart loop that could repeatedly fight permission revocation.
      EmitError(CGPreflightListenEventAccess() ? @"unavailable" : @"permission-required");
      CFRunLoopStop(CFRunLoopGetMain());
      return event;
    }
    bool capture = CommandGestureUpdate(info, type,
      (CGKeyCode)CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode),
      CGEventGetFlags(event), CGEventGetIntegerValueField(event, kCGKeyboardEventAutorepeat) != 0);
    if (capture) {
      pid_t pid = NSWorkspace.sharedWorkspace.frontmostApplication.processIdentifier;
      NSArray *windows = CFBridgingRelease(CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID));
      NSDictionary *target = CommandCaptureWindow(windows, pid);
      // The desktop or an app without a normal window is not a capture target.
      // A missed target does not broaden into a full-display capture.
      if (target) EmitJSON(target);
    }
    return event; // listen-only: do not consume or alter another app's input.
  }
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    signal(SIGPIPE, SIG_IGN);
    int stdoutFlags = fcntl(STDOUT_FILENO, F_GETFL);
    if (stdoutFlags >= 0) fcntl(STDOUT_FILENO, F_SETFL, stdoutFlags | O_NONBLOCK);
    bool check = argc == 2 && strcmp(argv[1], "--check") == 0;
    bool request = argc == 2 && strcmp(argv[1], "--request-permission") == 0;
    if (argc > 1 && !check && !request) { EmitError(@"unavailable"); return 64; }
    bool allowed = CGPreflightListenEventAccess();
    if (!allowed && request) allowed = CGRequestListenEventAccess();
    if (!allowed) { EmitError(@"permission-required"); return 2; }
    if (check) { EmitJSON(@{ @"type": @"ready" }); return 0; }

    CommandGesture gesture = {0};
    // One initial modifier snapshot, not key polling: a chord held before this
    // process started must finish before a new gesture can arm.
    CGEventFlags initial = CGEventSourceFlagsState(kCGEventSourceStateCombinedSessionState);
    gesture.left = (initial & NX_DEVICELCMDKEYMASK) != 0;
    gesture.right = (initial & NX_DEVICERCMDKEYMASK) != 0;
    gesture.blocked = (initial & kCGEventFlagMaskCommand) != 0;
    CGEventMask mask = CGEventMaskBit(kCGEventFlagsChanged)
      | CGEventMaskBit(kCGEventKeyDown) | CGEventMaskBit(kCGEventKeyUp);
    CFMachPortRef tap = CGEventTapCreate(kCGSessionEventTap, kCGHeadInsertEventTap,
      kCGEventTapOptionListenOnly, mask, ObserveEvent, &gesture);
    if (!tap) { EmitError(CGPreflightListenEventAccess() ? @"unavailable" : @"permission-required"); return 3; }
    CFRunLoopSourceRef source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0);
    if (!source) { CFMachPortInvalidate(tap); CFRelease(tap); EmitError(@"unavailable"); return 3; }
    CFRunLoopAddSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes);

    // The parent's pipe EOF and termination signals all take the same clean
    // teardown path. There are no timers or background key-state polling loops.
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
    dispatch_resume(term);
    dispatch_resume(interrupt);
    dispatch_resume(input);
    CGEventTapEnable(tap, true);
    EmitJSON(@{ @"type": @"ready" });
    CFRunLoopRun();
    dispatch_source_cancel(input);
    dispatch_source_cancel(interrupt);
    dispatch_source_cancel(term);
    CGEventTapEnable(tap, false);
    CFMachPortInvalidate(tap);
    CFRunLoopRemoveSource(CFRunLoopGetMain(), source, kCFRunLoopCommonModes);
    CFRelease(source);
    CFRelease(tap);
    return 0;
  }
}
#endif
