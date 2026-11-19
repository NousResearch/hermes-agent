# Authentication Patterns

Patterns for handling login flows, session persistence, and authenticated browsing using the Actionbook Rust CLI.

## Basic Login Flow

```bash
# Open login page
actionbook browser open "https://app.example.com/login"
actionbook browser wait-nav

# Take snapshot to identify form elements
actionbook browser snapshot

# Fill credentials using CSS selectors from snapshot
actionbook browser fill "#email" "$APP_USERNAME"
actionbook browser fill "#password" "$APP_PASSWORD"

# Submit
actionbook browser click "button[type=submit]"
actionbook browser wait-nav

# Verify login succeeded by checking the current URL
actionbook browser eval "window.location.href"
```

## Profile-Based Session Persistence

Profiles provide isolated browser sessions with separate cookies and storage. Use profiles to persist authentication across CLI invocations.

```bash
# Create a dedicated profile for the target site
actionbook profile create myapp

# Login using that profile
actionbook -P myapp browser open "https://app.example.com/login"
actionbook -P myapp browser wait-nav
actionbook -P myapp browser snapshot
actionbook -P myapp browser fill "#email" "$APP_USERNAME"
actionbook -P myapp browser fill "#password" "$APP_PASSWORD"
actionbook -P myapp browser click "button[type=submit]"
actionbook -P myapp browser wait-nav

# Later: reuse the profile — session cookies are preserved
actionbook -P myapp browser open "https://app.example.com/dashboard"
actionbook -P myapp browser wait-nav
actionbook -P myapp browser text "h1"  # Should show dashboard content

# List and inspect profiles
actionbook profile list
actionbook profile show myapp
```

## OAuth / SSO Flows

For OAuth redirects, use snapshot at each step to identify form elements:

```bash
# Start OAuth flow
actionbook browser open "https://app.example.com/auth/google"
actionbook browser wait-nav

# Google sign-in page — snapshot to find form elements
actionbook browser snapshot
actionbook browser fill "input[type=email]" "$GOOGLE_EMAIL"
actionbook browser click "#identifierNext"
actionbook browser wait-nav

# Password step
actionbook browser snapshot
actionbook browser fill "input[type=password]" "$GOOGLE_PASSWORD"
actionbook browser click "#passwordNext"
actionbook browser wait-nav

# Wait for redirect back to app (OAuth redirects can be slow)
actionbook browser eval "window.location.href"  # Should be app.example.com
```

## Two-Factor Authentication

Handle 2FA by running in headed mode (the default) and waiting for manual input:

```bash
# Login with credentials (browser is visible by default)
actionbook browser open "https://app.example.com/login"
actionbook browser wait-nav
actionbook browser snapshot
actionbook browser fill "#email" "$APP_USERNAME"
actionbook browser fill "#password" "$APP_PASSWORD"
actionbook browser click "button[type=submit]"
actionbook browser wait-nav

# Wait for user to complete 2FA manually
# Use a long timeout and wait for an element only present after 2FA
actionbook browser wait ".dashboard-header" --timeout 120000
```

## Cookie-Based Auth

Manually set authentication cookies:

```bash
# Open the target domain first (cookies need a page context)
actionbook browser open "https://app.example.com"
actionbook browser wait-nav

# Set auth cookie (use environment variable for the token value)
actionbook browser cookies set "session_token" "$SESSION_TOKEN" --domain ".example.com"

# Navigate to protected page
actionbook browser goto "https://app.example.com/dashboard"
actionbook browser wait-nav
```

## Token Refresh Handling

For sessions with expiring tokens:

```bash
#!/bin/bash
# Wrapper that handles token refresh using profiles

PROFILE="myapp"

# Try navigating directly with existing profile session
actionbook -P "$PROFILE" browser open "https://app.example.com/dashboard"
actionbook -P "$PROFILE" browser wait-nav

# Check if session is still valid via URL
URL=$(actionbook -P "$PROFILE" browser eval "window.location.href")
if [[ "$URL" == *"/login"* ]]; then
    echo "Session expired, re-authenticating..."

    # Perform fresh login
    actionbook -P "$PROFILE" browser snapshot
    actionbook -P "$PROFILE" browser fill "#email" "$APP_USERNAME"
    actionbook -P "$PROFILE" browser fill "#password" "$APP_PASSWORD"
    actionbook -P "$PROFILE" browser click "button[type=submit]"
    actionbook -P "$PROFILE" browser wait-nav
fi
```

## Security Best Practices

1. **Use environment variables for credentials** — source from a `.env` file, not inline
   ```bash
   # Load from .env file (avoid typing secrets in the terminal)
   source .env  # contains APP_USERNAME, APP_PASSWORD
   actionbook browser fill "#email" "$APP_USERNAME"
   actionbook browser fill "#password" "$APP_PASSWORD"
   ```

2. **Protect profile directories** — profiles store cookies and session data on disk
   ```bash
   # Profile data is equivalent to passwords — restrict access
   chmod 700 ~/.config/actionbook/
   ```

3. **Clean up sessions after automation**
   ```bash
   actionbook browser cookies clear
   actionbook browser close
   ```

4. **Delete profiles when no longer needed**
   ```bash
   actionbook profile delete myapp
   ```

5. **Use headless mode for CI/CD** (no visible browser)
   ```bash
   actionbook --headless browser open "https://app.example.com/login"
   # ... login and perform actions ...
   actionbook browser close
   ```
