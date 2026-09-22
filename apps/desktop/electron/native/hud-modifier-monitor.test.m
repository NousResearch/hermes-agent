#define HUD_MODIFIER_MONITOR_TEST 1
#include "hud-modifier-monitor.m"
#include <assert.h>

static CGEventFlags command = kCGEventFlagMaskCommand | NX_DEVICELCMDKEYMASK;
static CGEventFlags option = kCGEventFlagMaskAlternate | NX_DEVICELALTKEYMASK;
static bool flags(HudMacState *s, CGKeyCode key, CGEventFlags f, uint64_t ms) {
  return HudMacUpdate(s, kCGEventFlagsChanged, key, f, false, 0, ms);
}
static void begin(HudMacState *s) {
  assert(!flags(s, 55, command, 10));
  assert(!flags(s, 58, command | option, 20));
}
static bool release(HudMacState *s) {
  assert(!flags(s, 55, option, 30));
  return flags(s, 58, 0, 40);
}
int main(void) { @autoreleasepool {
  HudMacState s = {0};
  begin(&s);
  assert(release(&s));
  CGEventType interruptions[] = { kCGEventKeyDown, kCGEventKeyUp,
    kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGEventRightMouseDown,
    kCGEventOtherMouseDown, kCGEventScrollWheel, kCGEventLeftMouseDragged,
    kCGEventRightMouseDragged, kCGEventOtherMouseDragged };
  for (unsigned i = 0; i < sizeof(interruptions) / sizeof(*interruptions); i++) {
    s = (HudMacState){0};
    begin(&s);
    assert(!HudMacUpdate(&s, interruptions[i], 0, command | option, false, 0, 25));
    assert(!release(&s));
  }
  s = (HudMacState){0}; begin(&s);
  assert(!HudMacUpdate(&s, kCGEventKeyDown, 0, command | option, true, 0, 25));
  assert(!release(&s));
  s = (HudMacState){0}; begin(&s);
  assert(!flags(&s, 56, command | option | kCGEventFlagMaskShift, 25));
  assert(!flags(&s, 56, command | option, 26));
  assert(!release(&s));
  s = (HudMacState){0}; begin(&s);
  assert(!flags(&s, 54, command | option | NX_DEVICERCMDKEYMASK, 25));
  assert(!release(&s));
  // Plain pointer movement is harmless; drag/click/wheel was tested above.
  s = (HudMacState){0}; begin(&s);
  assert(!HudMacUpdate(&s, kCGEventMouseMoved, 0, command | option, false, 0, 25));
  assert(release(&s));
  // Snapshots of a held key, mouse, or chord cannot authorize a release.
  s = (HudMacState){0}; s.keys[0] = true; s.heldKeys = 1;
  begin(&s); assert(!release(&s));
  s = (HudMacState){0}; s.buttons = 1;
  begin(&s); assert(!release(&s));
  s = (HudMacState){0};
  HudModifierUpdate(&s.gesture, 3, true, true, 0);
  assert(!release(&s));
  // Two Command keys retain their independent screenshot meaning.
  s = (HudMacState){0};
  assert(!flags(&s, 55, command, 10));
  assert(!flags(&s, 54, command | NX_DEVICERCMDKEYMASK, 20));
  assert(!flags(&s, 54, command, 30));
  assert(!flags(&s, 55, 0, 40));
  puts("native macOS event adapter assertions passed");
} return 0; }
