# Bryan/Hermitage premium packet branding split — 2026-06-20

## Session lesson
A useful product-model distinction emerged during Sergio/Darin premium lesson packet work:

- **The System premium packet template** is the correct shared fallback for coaches who have not completed a brand design exercise.
- Once a coach has completed the brand exercise, the premium packet workflow should stop being purely global at the branding layer.

## Durable class-level rule
Keep a two-layer structure:

1. **Shared class-level lesson recap skill**
   - owns routing logic
   - decides robust lesson media -> premium packet
   - enforces artifact-vs-chat discipline
   - defines blocker behavior and fallback hierarchy

2. **Coach-specific branding/render layer**
   - owns post-brand-exercise brand assets, layout/voice constraints, renderer choices, and coach-specific defaults
   - should be loaded by the coach agent once the brand exercise is complete

This avoids two bad extremes:
- one giant global skill that tries to carry every coach's branding forever
- isolated coach skills that duplicate the routing logic and lose shared lessons

## Concrete Bryan/Darin implication
For Bryan:
- Darin should still use the shared `coach-lesson-recap-artifacts` routing skill.
- Bryan/Hermitage-specific premium output should eventually be driven by a Bryan-specific branding skill/config layer.
- Hermitage logos were staged locally during this session as prep evidence, but that alone does not prove the Bryan-branded renderer is wired.

## Truthfulness rule
Do not claim a coach-specific branded packet exists just because local logos or brand assets are present. Separate:
- shared premium packet capability
- coach-specific brand assets staged
- coach-specific renderer/path actually proven live
