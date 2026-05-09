# Performance and Debugging

Interactive 3D web work can look correct and still be unusable. Verification must include performance.

## Default performance rules

- cap DPR
- keep scene graphs small
- prefer fewer lights
- avoid expensive postprocessing until needed
- do not load oversized assets casually

## Start simple

Before optimizing:

- remove postprocessing
- reduce light count
- simplify geometry
- turn off controls if they are not required
- test with one asset only

## Console first

Always inspect browser console output.

Look for:

- WebGL context loss
- failed asset fetches
- shader compile errors
- hydration/runtime errors
- warnings caused by repeated mounts

## Visual symptoms and likely causes

- **Black canvas** -> camera, lighting, or shader failure
- **White/unstyled fallback only** -> client boundary not mounting
- **Jank on laptop fans** -> DPR too high or scene too heavy
- **Works once, breaks on navigation** -> cleanup leak
- **Model invisible** -> scale, framing, near/far planes, or asset origin issue

## Cleanup checklist

- stop animation loop
- dispose controls
- dispose geometries
- dispose materials
- dispose textures when custom-managed
- remove event listeners

## Mobile / lower-power posture

For public-facing surfaces:

- reduce DPR
- simplify materials
- reduce postprocessing
- provide a graceful fallback

If the page's value survives without 3D, the fallback should still feel intentional.
