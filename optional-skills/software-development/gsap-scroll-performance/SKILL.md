---
name: gsap-scroll-performance
description: "Fix GSAP + Lenis scroll performance in React: blank hero on load, slow scrolling, sluggish ScrollTrigger animations."
tags: [gsap, lenis, scroll, animation, performance, react, vite, frontend]
triggers:
  - "GSAP scroll animation feels slow or laggy"
  - "Page loads blank, content only appears after scrolling"
  - "Lenis smooth scroll is too slow"
  - "ScrollTrigger pinned sections feel sticky or sluggish"
  - "Hero section hidden until user scrolls"
  - "blank page on load with GSAP animations"
---

# GSAP + Lenis Scroll Performance Tuning

When a React landing page uses GSAP ScrollTrigger + Lenis smooth scroll and the user reports blank content on load, slow scrolling, or sluggish animations, apply these fixes in order.

## Diagnosis Checklist

1. **Blank on load?** — Check if hero elements have `opacity-0` in CSS AND use `scrub` ScrollTrigger. The scrub delays opacity until scroll, hiding content.
2. **Slow scrolling?** — Check Lenis `duration` (default 1.2 is too slow). Also check for missing `wheelMultiplier`/`touchMultiplier`.
3. **Sluggish pinned sections?** — Check `scrub` values (1.0 is smooth but slow, 0.5 is snappier). Check `start` positions.

## Fix 1: Hero Blank on Load

**Root cause**: Hero uses `scrub: 1` + `fromTo({opacity: 0})` in a ScrollTrigger. Content stays invisible until the user scrolls.

**Fix**: Replace hero ScrollTrigger with a mount fade-in animation:

```tsx
// ❌ WRONG — hero invisible until scroll
const heroTl = gsap.timeline({
  scrollTrigger: {
    trigger: heroRef.current,
    start: 'top top',
    end: '+=800',
    pin: true,
    scrub: 1,        // ← this hides content until scroll
  },
});
heroTl.fromTo(heroTitle, { opacity: 0, y: 60 }, { opacity: 1, y: 0 });

// ✅ CORRECT — hero visible on mount with subtle animation
if (heroTitle) gsap.set(heroTitle, { y: 50 });  // initial offset
if (heroCta) gsap.set(heroCta, { y: 30, scale: 0.95 });

const heroTl = gsap.timeline({ delay: 0.2 });
heroTl.to(heroTitle, { opacity: 1, y: 0, duration: 0.7, ease: 'power2.out' }, 0.1);
heroTl.to(heroCta, { opacity: 1, y: 0, scale: 1, duration: 0.5, ease: 'power2.out' }, 0.5);
```

**Key points**:
- Keep `opacity-0` in CSS className (prevents FOUC before GSAP loads)
- Use `gsap.set()` for initial y/scale offsets
- Use `gsap.to()` to animate TO visible state
- Sequence with staggered timings (0.1s, 0.25s, 0.5s)
- If hero has multiple `.hero-subtitle` elements, use `querySelectorAll` and forEach
- Do NOT pin the hero section — it should scroll naturally

## Fix 2: Slow Scrolling (Lenis)

**Root cause**: Lenis `duration` too high, missing multipliers.

```tsx
// ❌ TOO SLOW
const lenis = new Lenis({
  duration: 1.2,
  smoothWheel: true,
});

// ✅ SNAPPY
const lenis = new Lenis({
  duration: 0.8,
  easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
  smoothWheel: true,
  wheelMultiplier: 1.2,
  touchMultiplier: 1.5,
});
```

## Fix 3: ScrollTrigger Performance

For **pinned sections** (horizontal scroll, step-by-step reveals):
- Reduce `scrub: 1` → `scrub: 0.5` for snappier response
- Change `start: 'top top'` → `start: 'top 80%'` so animation triggers earlier
- Do NOT remove scrub from pinned sections — they need it for pin behavior

For **non-pinned scroll animations** (fade-in on scroll):
- Add `once: true` so animations don't re-trigger on scroll up
- Add `start: 'top 80%'` for earlier triggering
- Prefer framer-motion `useInView({ once: true })` over ScrollTrigger for simple fade-ins

## Lenis + GSAP Integration Pattern

```tsx
useEffect(() => {
  const lenis = new Lenis({ /* config */ });
  lenis.on('scroll', ScrollTrigger.update);
  gsap.ticker.add((time: number) => { lenis.raf(time * 1000); });
  gsap.ticker.lagSmoothing(0);

  // ... ScrollTrigger setup ...

  return () => {
    ScrollTrigger.getAll().forEach((t) => t.kill());
    lenis.destroy();
  };
}, []);
```

## Pitfalls

- **Hero pinned + scrub = blank page**: The #1 cause of "page loads blank" with GSAP. Hero content should never depend on scroll to become visible.
- **scrub on hero = bad UX**: Users expect to see the hero immediately. Pin+scrub is for mid-page sections only.
- **duration > 1.0 on Lenis**: Feels like wading through mud. 0.8 is the sweet spot.
- **Missing wheelMultiplier**: Without it, trackpad and mouse wheel feel sluggish compared to keyboard/touch.
- **once: true doesn't work with scrub**: `once` is for trigger-based animations, not scrub-linked ones. Pinned sections with scrub always need scrub.
- **querySelector vs querySelectorAll for hero**: If multiple elements share a class (e.g., two `.hero-subtitle` spans), `querySelector` only gets the first. Use `querySelectorAll` + `forEach`.
