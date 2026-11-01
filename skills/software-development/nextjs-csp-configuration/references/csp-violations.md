# Real-World CSP Violation Transcripts

Common CSP errors seen in Next.js apps and their fixes.

## 1. Inline Script Blocked

```
Refused to execute inline script because it violates the following
Content Security Policy directive: "script-src 'self'"
```

**Fix:** Use `'unsafe-inline'` (dev only) or nonce-based CSP:
```typescript
const nonce = crypto.randomUUID();
// Add nonce to script tag or use Next.js middleware
```

## 2. External Media Blocked

```
Refused to load the image because it violates the following
Content Security Policy directive: "img-src 'self'"
```

**Fix:** Add the CDN to `img-src`:
```
img-src 'self' https://d123.cloudfront.net https://*.twimg.com
```

## 3. Connect-src for API Calls

```
Refused to connect to because it violates the following
Content Security Policy directive: "connect-src 'self'"
```

**Fix:** Add your API domains:
```
connect-src 'self' https://api.example.com https://*.supabase.co
```

## 4. Style-src for External Fonts

```
Refused to apply inline style because it violates the following
Content Security Policy directive: "style-src 'self'"
```

**Fix:** Allow Google Fonts:
```
style-src 'self' 'unsafe-inline' https://fonts.googleapis.com
font-src 'self' https://fonts.gstatic.com
```

## 5. Frame-src for Embeds

```
Refused to frame because it violates the following
Content Security Policy directive: "frame-src 'self'"
```

**Fix:** Allow YouTube embeds:
```
frame-src 'self' https://www.youtube.com https://www.youtube-nocookie.com
```
