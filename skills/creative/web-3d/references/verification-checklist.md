# Verification Checklist

Use this after implementation. Do not stop at "it compiled."

## Visual

- [ ] The scene appears immediately
- [ ] The composition matches the brief
- [ ] The 3D surface belongs to the page instead of fighting it
- [ ] The camera framing is intentional
- [ ] Lighting is readable and not accidental

## Behavior

- [ ] Resize works
- [ ] Interactions work
- [ ] Controls, if present, feel justified
- [ ] No duplicate canvases or ghost surfaces appear after navigation/remount

## Console

- [ ] No uncaught runtime errors
- [ ] No failed asset fetches
- [ ] No WebGL/shader warnings that indicate broken output

## Integration

- [ ] No SSR or hydration crash path
- [ ] Asset paths work in the actual project structure
- [ ] The 3D surface respects existing layout and design constraints

## Cleanup

- [ ] Unmount/remount does not leak or duplicate
- [ ] Resources are disposed when appropriate

## Performance

- [ ] Acceptable on the target machine/surface
- [ ] DPR is capped when needed
- [ ] Scene complexity is justified
- [ ] There is a fallback if the surface is public-facing
