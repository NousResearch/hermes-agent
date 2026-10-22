#include "hud-modifier-gesture.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
  // Either press/release ordering is valid, but neither first release summons.
  for (unsigned first = 1; first <= 2; first++) {
    for (unsigned last = 1; last <= 2; last++) {
      HudModifierGesture s = {0};
      assert(!HudModifierUpdate(&s, first, false, false, 10));
      assert(!HudModifierUpdate(&s, 3, false, false, 20));
      assert(!HudModifierUpdate(&s, last, false, false, 30));
      assert(HudModifierUpdate(&s, 0, false, false, 40));
      assert(!HudModifierUpdate(&s, 0, false, false, 50));
    }
  }
  // A foreign event at any phase permanently cancels this attempt, even if
  // the foreign key/button is released before the target modifiers.
  for (int phase = 0; phase < 3; phase++) {
    HudModifierGesture s = {0};
    HudModifierUpdate(&s, 1, false, phase == 0, 10);
    HudModifierUpdate(&s, 3, false, phase == 1, 20);
    HudModifierUpdate(&s, 2, false, phase == 2, 30);
    assert(!HudModifierUpdate(&s, 0, false, false, 40));
    HudModifierUpdate(&s, 1, false, false, 50);
    HudModifierUpdate(&s, 3, false, false, 60);
    HudModifierUpdate(&s, 2, false, false, 70);
    assert(HudModifierUpdate(&s, 0, false, false, 80));
  }
  // Extra modifiers, held keys/buttons, and partial-release re-press cannot arm.
  for (int kind = 0; kind < 3; kind++) {
    HudModifierGesture s = {0};
    HudModifierUpdate(&s, 1, kind == 1, false, 10);
    HudModifierUpdate(&s, kind == 0 ? 7 : 3, kind == 1, false, 20);
    if (kind == 0) HudModifierUpdate(&s, 3, false, false, 25);
    HudModifierUpdate(&s, 2, false, false, 30);
    if (kind == 2) {
      HudModifierUpdate(&s, 3, false, false, 35);
      HudModifierUpdate(&s, 2, false, false, 36);
    }
    assert(!HudModifierUpdate(&s, 0, false, false, 40));
  }
  for (unsigned initial = 1; initial <= 3; initial++) {
    HudModifierGesture s = {0};
    HudModifierUpdate(&s, initial, true, true, 0);
    HudModifierUpdate(&s, 3, false, false, 10);
    HudModifierUpdate(&s, 2, false, false, 20);
    assert(!HudModifierUpdate(&s, 0, false, false, 30));
  }
  for (uint64_t duration = 500; duration <= 501; duration++) {
    HudModifierGesture s = {0};
    HudModifierUpdate(&s, 1, false, false, 10);
    HudModifierUpdate(&s, 3, false, false, 20);
    HudModifierUpdate(&s, 2, false, false, 30);
    assert(HudModifierUpdate(&s, 0, false, false, 10 + duration) == (duration == 500));
  }
  puts("native modifier gesture assertions passed");
  return 0;
}
