---
name: Next.js Content-Security-Policy Configuration
title: Next.js Content-Security-Policy Configuration
description: "Content-Security-Policy headers for Next.js apps."
tags: [nextjs, security, csp, headers, media, cdn, debugging]
---

# Next.js Content-Security-Policy Configuration

## When to Use This Skill

- External media (audio/video) fails to load with CSP violations
- CDN resources (images, fonts, media) are blocked
- Need to allow specific external domains in CSP
- Debugging "Content Security Policy directive violates" errors in browser console
- Configuring security headers in `next.config.ts`

## Quick Reference: CSP Directives

| Directive | Purpose | Common Sources |
|-----------|---------|----------------|
| `default-src` | Fallback for all directives | `'self'` (localhost + production domain) |
| `media-src` | Audio and video | `'self'`, `https://cdn.example.com` |
| `img-src` | Images | `'self'`, `blob:`, `data:`, CDNs |
| `script-src` | JavaScript | `'self'`, `'unsafe-eval'`, `'unsafe-inline'` (dev only) |
| `style-src` | CSS | `'self'`, `'unsafe-inline'`, `https://fonts.googleapis.com` |
| `font-src` | Fonts | `'self'`, `https://fonts.gstatic.com` |
| `connect-src` | Fetch/AJAX/WebSocket | `'self'`, API domains, Sentry, Supabase |

## Standard Next.js CSP Template

```typescript
// next.config.ts
import type { NextConfig } from 'next';

const ContentSecurityPolicy = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-eval' 'unsafe-inline'", // Required for Next.js dev mode
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' https://fonts.gstatic.com",
  "img-src 'self' blob: data: https://*.supabase.co",
  "media-src 'self' https://your-cdn.example.com", // Add external media sources here
  "connect-src 'self' https://*.supabase.co https://o377792.ingest.us.sentry.io",
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join('; ');

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
          { key: 'X-Powered-By', value: '' },
          { key: 'Content-Security-Policy', value: ContentSecurityPolicy },
        ],
      },
    ];
  },
};

export default nextConfig;
```

## Common Pitfalls

### External Media Blocked by CSP

**Symptom**: Audio/video player renders but shows error in console:
```
Loading media from 'https://cdn.example.com/audio.mp3' violates the following
Content Security Policy directive: "default-src 'self'"
```

**Cause**: `default-src 'self'` blocks all external resources. Media-specific directive missing.

**Fix**: Add `media-src` directive:
```typescript
"media-src 'self' https://cdn.example.com"
```

### Biome A11y Rule Conflicts

**Symptom**: Biome linter errors on `<audio>` or `<video>` elements:
```
lint/a11y/useMediaCaption: Provide a track for captions when using audio or video elements
```

**Cause**: Accessibility rules require captions, but external media may not have transcripts.

**Fix**: Suppress the rule (NOT `mediaHasCaption` - that's wrong syntax):
```tsx
{/* biome-ignore lint/a11y/useMediaCaption: External media without available captions */}
<audio controls preload="none">
  <source src={audioUrl} type="audio/mpeg" />
</audio>
```

### Wrong Biome Suppression Syntax

**Incorrect**:
```tsx
// biome-ignore lint/a11y/mediaHasCaption  // ❌ Wrong rule name
```

**Correct**:
```tsx
// biome-ignore lint/a11y/useMediaCaption  // ✅ Correct rule name
```

### CSP Changes Require Server Restart

**Symptom**: Modified `next.config.ts` but CSP violations persist after page refresh.

**Cause**: CSP headers are cached at server startup. Changes in `next.config.ts` require restart.

**Fix**: Kill and restart dev server:
```bash
# Find and kill process
pkill -f "next dev"

# Restart
npm run dev
```

## Debugging Steps

1. **Check browser console for CSP violations**:
   - Open DevTools → Console
   - Look for "Content Security Policy" errors
   - Note which directive is violated and which resource is blocked

2. **Verify CSP headers in response**:
   ```bash
   curl -I http://localhost:3000/your-page
   # Look for: content-security-policy: header
   ```

3. **Test external resource accessibility**:
   ```bash
   # Verify CDN returns proper headers
   curl -I https://cdn.example.com/audio.mp3
   # Should return: HTTP/2 200, content-type: audio/mpeg
   ```

4. **Check component rendering** (if audio appears but doesn't work):
   ```javascript
   // In browser console
   const audio = document.querySelector('audio');
   const source = audio?.querySelector('source');
   console.log({
     src: source?.src,
     canPlay: audio?.canPlayType('audio/mpeg'),
     error: audio?.error
   });
   ```

5. **Restart dev server after CSP config changes**:
   ```bash
   pkill -f "next dev"
   npm run dev
   ```

## Testing Patterns

### Test Audio Loading (Vitest + Testing Library)

```tsx
// Mock external audio URLs in tests
const mockDays = [
  {
    id: '01',
    title: 'Lesson 1',
    date: '09/05/2026',
    audio: 'https://cdn.example.com/audio.mp3'  // Include audio in mock data
  }
];

// Test that audio element renders with correct src
expect(screen.getByTestId('audio-element')).toBeInTheDocument();
const audio = screen.getByTestId('audio-element');
expect(audio.querySelector('source')).toHaveAttribute('src', 'https://cdn.example.com/audio.mp3');
```

### Test Button Disabled State (aria-disabled)

```tsx
// Use aria-disabled attribute instead of toBeDisabled()
const btn = screen.getByTestId('btn-complete');
expect(btn).toHaveAttribute('aria-disabled', 'true');
expect(btn).toHaveTextContent('Completado');
```

## External Media Sources (Examples)

| Service | Base URL | Media Pattern |
|---------|----------|---------------|
| Novo Tempo (Brazil) | `https://vod.novotempo.org.br` | `/mp3/LicoesDaBiblia2019/LDB{DD}-{MM}-{YYYY}.mp3` |
| CloudFront | `https://d123.cloudfront.net` | `/media/audio/*.mp3` |
| Vercel Blob | `https://*.public.blob.vercel-storage.com` | `/*` |

Add these to `media-src` as needed:
```typescript
"media-src 'self' https://vod.novotempo.org.br https://d123.cloudfront.net"
```

## References

- [MDN: Content-Security-Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)
- [Next.js: Custom Headers](https://nextjs.org/docs/app/building-your-application/configuring/headers)
- [Biome: A11y Rules](https://biomejs.dev/linter/rules/a11y/)
- See `references/csp-violations.md` for real-world error transcripts
