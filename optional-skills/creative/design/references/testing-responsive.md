# Testing Responsive Design

A design is not done until it is verified on multiple sizes and input methods.

## Browser DevTools

- Use device emulation for quick checks.
- Test at 320px, 375px, 768px, 1024px, 1440px, and 1920px.
- Rotate to landscape on small screens.
- Test with the device pixel ratio set to 2x or 3x.

## Real Devices

- Test on at least one iOS and one Android device.
- Check touch targets with a real thumb.
- Verify keyboard input on tablets and desktop.
- Confirm that reduced-motion is respected.

## What to Check

- [ ] Layout does not break at any breakpoint.
- [ ] Text is readable without zoom.
- [ ] Touch targets are easy to hit.
- [ ] Forms work with on-screen and hardware keyboards.
- [ ] Images load correctly and do not overflow.
- [ ] Focus is visible and follows a logical order.
- [ ] Color contrast passes WCAG AA.

## Network and Performance

- Test on a slow 3G or Fast 3G connection.
- Confirm images are sized appropriately.
- Check for layout shift (CLS) during load.
