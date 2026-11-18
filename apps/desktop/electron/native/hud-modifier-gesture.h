#ifndef HERMES_HUD_MODIFIER_GESTURE_H
#define HERMES_HUD_MODIFIER_GESTURE_H
#include <stdbool.h>
#include <stdint.h>

// Only a summary of the current gesture survives an event; no input history.
// Bits 1/2 are the two target modifiers; bit 4 means any extra modifier.
typedef struct {
  unsigned modifiers;
  bool started;
  bool both;
  bool blocked;
  bool releasing;
  uint64_t began;
} HudModifierGesture;

static inline bool HudModifierUpdate(HudModifierGesture *s, unsigned modifiers,
                                      bool held, bool interrupted, uint64_t now) {
  if (held || interrupted || (modifiers & ~3u)) s->blocked = true;
  unsigned pressed = modifiers & ~s->modifiers;
  if (s->releasing && pressed) s->blocked = true;
  if (s->modifiers & ~modifiers) s->releasing = true;
  if (!s->modifiers && !s->blocked && (modifiers == 1 || modifiers == 2)) {
    s->started = true;
    s->began = now;
  }
  if (modifiers && !s->started) s->blocked = true;
  if (modifiers == 3 && s->started) s->both = true;
  bool summon = !modifiers && s->both && !s->blocked
    && now >= s->began && now - s->began <= 500;
  s->modifiers = modifiers;
  if (!modifiers && !held) *s = (HudModifierGesture){0};
  return summon;
}
#endif
