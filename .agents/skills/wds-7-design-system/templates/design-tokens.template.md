# Design Tokens

**Last Updated:** [Date]
**Token Count:** [count]

---

## Colors

### Primary Colors

```yaml
primary-50: #eff6ff
primary-100: #dbeafe
primary-200: #bfdbfe
primary-300: #93c5fd
primary-400: #60a5fa
primary-500: #3b82f6
primary-600: #2563eb
primary-700: #1d4ed8
primary-800: #1e40af
primary-900: #1e3a8a
```

### Semantic Colors

```yaml
success: #10b981
error: #ef4444
warning: #f59e0b
info: #3b82f6
```

### Neutral Colors

```yaml
gray-50: #f9fafb
gray-100: #f3f4f6
[... more grays]
gray-900: #111827
```

---

## Typography

### Font Families

```yaml
font-sans: 'Inter, system-ui, sans-serif'
font-mono: 'JetBrains Mono, monospace'
```

### Font Sizes

```yaml
text-xs: 0.75rem
text-sm: 0.875rem
text-base: 1rem
text-lg: 1.125rem
text-xl: 1.25rem
text-2xl: 1.5rem
text-3xl: 1.875rem
text-4xl: 2.25rem
```

### Font Weights

```yaml
font-normal: 400
font-medium: 500
font-semibold: 600
font-bold: 700
```

---

## Spacing

```yaml
spacing-0: 0
spacing-1: 0.25rem
spacing-2: 0.5rem
spacing-3: 0.75rem
spacing-4: 1rem
spacing-6: 1.5rem
spacing-8: 2rem
spacing-12: 3rem
spacing-16: 4rem
```

---

## Layout

### Breakpoints

```yaml
sm: 640px
md: 768px
lg: 1024px
xl: 1280px
2xl: 1536px
```

### Container Widths

```yaml
container-sm: 640px
container-md: 768px
container-lg: 1024px
container-xl: 1280px
```

---

## Effects

### Shadows

```yaml
shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05)
shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1)
shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1)
```

### Border Radius

```yaml
radius-sm: 0.125rem
radius-md: 0.375rem
radius-lg: 0.5rem
radius-full: 9999px
```

### Transitions

```yaml
transition-fast: 150ms
transition-base: 200ms
transition-slow: 300ms
```

---

## Component-Specific Tokens

### Button

```yaml
button-padding-x: spacing-4
button-padding-y: spacing-2
button-border-radius: radius-md
button-font-weight: font-semibold
```

### Input

```yaml
input-height: 2.5rem
input-padding-x: spacing-3
input-border-color: gray-300
input-border-radius: radius-md
```

---

**Tokens are automatically populated as components are added to the design system.**
