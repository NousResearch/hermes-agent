# CSS Frameworks

Use when the project will ship with a CSS framework and the user has not yet chosen one. Bootstrap and Tailwind are the two most common choices; both can be mobile-first, but the design process differs.

## How to choose

| Concern | Bootstrap | Tailwind CSS |
|---------|-----------|--------------|
| Speed of first screen | fast | fast after config |
| Custom look | requires overriding variables + utility overrides | easier to make distinctive |
| Team discipline | lower; components carry you | higher; class composition can spiral |
| Bundle size | larger if every component is imported | smaller with PurgeCSS/JIT |
| Prototyping | excellent | excellent once tokens are set |
| Maintainability | stable, versioned components | depends on token discipline |

**Rule of thumb**

- Choose **Bootstrap** when the team wants a proven component library and a fast start, and is willing to override the default look.
- Choose **Tailwind** when the design needs to feel custom, the team has tokens and a scale, and the UI is component-driven.

The framework choice does not change the mobile-first design order. It only changes how tokens become code.

## Bootstrap 5 example

A mobile-first pricing card starting at 375px and scaling to a 3-column grid.

```html
<div class="container py-5">
  <div class="row g-4 justify-content-center">
    <div class="col-12 col-md-6 col-lg-4">
      <div class="card h-100 shadow-sm">
        <div class="card-body">
          <h5 class="card-title text-primary">Starter</h5>
          <p class="display-6 fw-bold">$19</p>
          <p class="card-text text-muted">Perfect for a single project.</p>
          <ul class="list-unstyled mb-4">
            <li class="mb-2">5 projects</li>
            <li class="mb-2">1 user</li>
            <li class="mb-2">Email support</li>
          </ul>
          <a href="#" class="btn btn-primary w-100">Get started</a>
        </div>
      </div>
    </div>
    <!-- repeat for Pro and Enterprise -->
  </div>
</div>
```

### Mobile-first notes

- `.col-12` forces one column at the base size.
- `.col-md-6` and `.col-lg-4` add columns only when there is enough room.
- The card is `h-100` so equal-height cards on desktop keep the same mental model.
- `w-100` on the CTA makes the touch target easy to hit at 375px.

## Tailwind CSS example

Same pricing card in Tailwind. Tokens are explicit and the responsive prefixes drive the layout.

```html
<section class="py-12">
  <div class="container mx-auto px-4">
    <div class="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      <article class="flex flex-col rounded-2xl border border-slate-200 p-6 shadow-sm">
        <h3 class="text-lg font-semibold text-blue-600">Starter</h3>
        <p class="my-2 text-4xl font-bold text-slate-900">$19</p>
        <p class="mb-4 text-slate-600">Perfect for a single project.</p>
        <ul class="mb-6 space-y-2 text-slate-700">
          <li>5 projects</li>
          <li>1 user</li>
          <li>Email support</li>
        </ul>
        <a href="#" class="mt-auto w-full rounded-lg bg-blue-600 px-4 py-3 text-center font-medium text-white hover:bg-blue-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-600">
          Get started
        </a>
      </article>
      <!-- repeat for Pro and Enterprise -->
    </div>
  </div>
</section>
```

### Mobile-first notes

- The `grid` starts as one column; `md:grid-cols-2` and `lg:grid-cols-3` add columns.
- `mt-auto` on the button pushes it to the bottom in a flex column.
- `focus-visible:outline` keeps keyboard focus visible without forcing mouse outlines.
- Spacing and type are explicit: no default framework theme decides the brand.

## Token mapping

Whichever framework is chosen, the design tokens must be set first.

| Token | Bootstrap | Tailwind |
|-------|-----------|----------|
| Colors | CSS variables in `:root` or SASS overrides | `tailwind.config.js` theme.extend.colors |
| Type scale | `$h1-font-size` etc. or custom CSS | `fontSize` and `lineHeight` config |
| Spacing | `$spacer` map, utility classes | `spacing` scale in config |
| Breakpoints | `$grid-breakpoints` | `screens` in config |
| Border radius | `$border-radius` variables | `borderRadius` config |

Do not ship Bootstrap with no `$theme-colors` override. Do not ship Tailwind with the default `blue-600` as the brand.

## Common mistakes

- Using a framework as the visual identity. The brand and the framework are not the same.
- Mixing Bootstrap and Tailwind in the same project without an explicit reason.
- Hiding the mobile layout behind too many breakpoints. Start at 375px, then add.
- Forgetting to override the default type scale. If the body is 14px or 0.875rem on mobile, the design will fail the quality floor.

## Accessibility

- Bootstrap: rely on `aria-*` and `role` attributes from components, but verify they are correct for the screen reader path.
- Tailwind: add `focus-visible:`, `sr-only`, and `aria-current` explicitly; nothing ships by default.
- Verify color contrast for both with the same values the brand tokens define.
