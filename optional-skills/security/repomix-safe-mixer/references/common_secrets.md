# Common Secret Patterns Reference

This document catalogs common credential types detected by the security scanner.

## Table of Contents

- [Cloud Provider Credentials](#cloud-provider-credentials)
- [Database Credentials](#database-credentials)
- [API Keys and Tokens](#api-keys-and-tokens)
- [Authentication Secrets](#authentication-secrets)
- [Common False Positives](#common-false-positives)

---

## Cloud Provider Credentials

### AWS Credentials

**AWS Access Key ID**:
- Pattern: `AKIA[0-9A-Z]{16}`
- Example: `AKIAIOSFODNN7EXAMPLE`
- Location: Often in `.env`, config files, or infrastructure code
- Risk: Full AWS account access

**AWS Secret Access Key**:
- Pattern: `[0-9a-zA-Z/+=]{40}`
- Context: Usually follows `aws_secret` or similar variable names
- Example: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`
- Risk: Account compromise, data breach, cost abuse

### Cloudflare R2

**R2 Account ID**:
- Pattern: `[0-9a-f]{32}` (in R2 URLs)
- Example: `89ff427005e1767943b5ac257905a280` in `https://89ff427005e1767943b5ac257905a280.r2.cloudflarestorage.com`
- Risk: Account identification, targeted attacks

**R2 Access Keys**:
- Similar to AWS S3 credentials
- Pattern: Standard access key + secret key pair
- Risk: Bucket access, file manipulation, cost abuse

---

## Database Credentials

### Supabase

**Project URL**:
- Pattern: `https://[a-z]{20}.supabase.co`
- Example: `https://ghyttjckzmzdxumxcixe.supabase.co`
- Risk: Project identification

**Anon/Public Key**:
- Pattern: JWT token starting with `eyJ`
- Example: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
- Risk: Public data access, edge function invocation, quota abuse

**Service Role Key**:
- Pattern: JWT token (longer than anon key)
- Risk: **CRITICAL** - Full database admin access, bypasses RLS

### PostgreSQL

**Connection String**:
- Pattern: `postgresql://user:password@host:port/database`
- Risk: Direct database access

---

## API Keys and Tokens

### Stripe

**Publishable Key**:
- Pattern: `pk_(live|test)_[0-9a-zA-Z]{24,}`
- Example: `pk_live_51AbC...` (truncated for security)
- Risk: Low (public by design, but reveals account)

**Secret Key**:
- Pattern: `sk_(live|test)_[0-9a-zA-Z]{24,}`
- Example: `sk_live_51AbC...` (truncated for security)
- Risk: **CRITICAL** - Payment processing, refunds, customer data

### OpenAI / Gemini / LLM Providers

**OpenAI API Key**:
- Pattern: `sk-[A-Za-z0-9]{48}`
- Risk: API abuse, cost accumulation

**Google Gemini API Key**:
- Pattern: `AIza[0-9A-Za-z_-]{35}`
- Risk: API abuse, quota exhaustion

**OpenRouter API Key**:
- Pattern: `sk-or-v1-[0-9a-f]{64}`
- Risk: API abuse via OpenRouter

### Cloudflare Turnstile

**Site Key**:
- Pattern: `0x[0-9A-F]{22}`
- Example: `0x4AAAAAABvH03QZ3BpnHR7p`
- Risk: Low (public by design), but enables testing

**Secret Key**:
- Pattern: `0x[0-9A-F]{40}`
- Risk: Bot protection bypass

---

## Authentication Secrets

### JWT Tokens

**Format**:
- Pattern: `eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+`
- Three base64url-encoded parts separated by dots
- Risk: Session hijacking, impersonation

### OAuth Secrets

**Client Secret**:
- Pattern: Variable, often `[0-9a-zA-Z_-]{20,}`
- Context: Near `client_id`, `oauth`, `app_secret`
- Risk: Application impersonation

### Private Keys

**RSA/EC Private Keys**:
- Pattern: `-----BEGIN (RSA|EC|OPENSSH|DSA) PRIVATE KEY-----`
- Risk: **CRITICAL** - Complete identity compromise

---

## Common False Positives

### Example/Placeholder Values

Safe to ignore when matching:
- Strings containing: `example`, `placeholder`, `test`, `demo`, `sample`
- Template variables: `<YOUR_API_KEY>`, `${API_KEY}`, `${...}`
- Documentation examples: `xxx`, `yyy`, `zzz`
- TODO markers: `TODO`, `FIXME`, `CHANGEME`

### Environment Variable References

Safe patterns (these are correct usage):
```javascript
// JavaScript/TypeScript
const apiKey = process.env.API_KEY;
const apiKey = import.meta.env.VITE_API_KEY;

// Python
api_key = os.getenv('API_KEY')
api_key = os.environ.get('API_KEY')

// Deno
const apiKey = Deno.env.get('API_KEY');
```

### Comments

Lines starting with comment markers are often documentation:
- `//` - JavaScript/TypeScript
- `#` - Python/Shell/YAML
- `/* */` - Multi-line comments

---

## Detection Strategies

### Context-Aware Scanning

Look for credential indicators:
- Variable names: `API_KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `PRIVATE_KEY`
- Assignment operators: `=`, `:`, `=>`
- Quote patterns: `"..."`, `'...'`, `` `...` ``

### File Type Priorities

**High Risk**:
- `.env`, `.env.local`, `.env.production`
- Configuration files: `config.json`, `settings.py`
- Infrastructure code: `.tf`, `.yaml` (Terraform, K8s)

**Medium Risk**:
- Source code: `.js`, `.ts`, `.py`, `.go`
- Documentation: `.md` (may contain examples)

**Low Risk**:
- Test files: `*.test.js`, `*.spec.ts`
- Example files: `*.example.*`

---

## Remediation Patterns

### Convert to Environment Variables

**Before** (hardcoded):
```javascript
const SUPABASE_URL = "https://ghyttjckzmzdxumxcixe.supabase.co";
const SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...";
```

**After** (environment variables):
```javascript
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || "https://your-project-ref.supabase.co";
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || "your-anon-key-here";

// Validation
if (!import.meta.env.VITE_SUPABASE_URL) {
  console.error("Missing VITE_SUPABASE_URL environment variable");
}
```

### Create .env.example

```bash
# Supabase Configuration
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your-anon-key-here

# API Keys
GEMINI_API_KEY=your-gemini-key
OPENROUTER_API_KEY=your-openrouter-key

# Important: Copy this to .env and replace with real values
# Never commit .env to version control!
```

---

## Post-Exposure Actions

If credentials are exposed:

1. **Rotate Immediately** - Generate new credentials
2. **Revoke Old Credentials** - Disable compromised keys
3. **Audit Usage** - Check for unauthorized access
4. **Monitor** - Set up alerts for unusual activity
5. **Update Code** - Deploy with new credentials
6. **Notify** - If public exposure, notify security team

---

## References

- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [AWS Credentials Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [GitHub Secret Scanning Patterns](https://docs.github.com/en/code-security/secret-scanning/about-secret-scanning)
