# Email Alias Login

An **opt-in bundled Hermes plugin** for mailboxes whose IMAP/SMTP login differs from the public email address. It re-registers the existing `email` gateway platform only in profiles that enable it. The bundled Email adapter still owns message parsing, sender authorization, threading, attachments, TLS and lifecycle; this plugin changes the authentication seams and standalone/cron sender without patching the adapter or core files.

This source is part of [hermes-agent PR #128161](https://github.com/NousResearch/hermes-agent/pull/128161), carrying the use case first proposed in [PR #77384](https://github.com/NousResearch/hermes-agent/pull/77384) and reported in [#41331](https://github.com/NousResearch/hermes-agent/issues/41331) and [#46676](https://github.com/NousResearch/hermes-agent/issues/46676). During the feature freeze, the plugin is the proposed upstream artifact; the old in-tree adapter patch is not the merge path.

## Find and enable

Once the source PR is merged into a Hermes installation, the plugin is present but **disabled by default**:

```sh
hermes plugins list
hermes plugins enable email-alias-login
```

Enable it in each profile whose mailbox needs a separate login. Other profiles continue using the bundled Email adapter. This plugin does not create a second email channel, change stored passwords, authorize new senders, or send mail at activation. A catalog listing can be pinned to an upstream commit **after the source PR merges**; the catalog cannot pin an unmerged commit to itself.

## Configure

Keep the public address and password in the profile's `.env`:

```dotenv
EMAIL_ADDRESS=alias@yourdomain.example
EMAIL_LOGIN_USER=account@provider.example
EMAIL_PASSWORD=<your-app-password>
EMAIL_IMAP_HOST=imap.provider.example
EMAIL_SMTP_HOST=smtp.provider.example
```

`EMAIL_LOGIN_USER` is optional. A `config.yaml` alternative is:

```yaml
plugins:
  enabled:
    - email-alias-login
platforms:
  email:
    enabled: true
    login_user: account@provider.example
```

The plugin manifest marks the password for a masked prompt and requires both IMAP and SMTP hosts. Resolution is **scoped `EMAIL_LOGIN_USER` → `platforms.email.login_user` → the public address**; whitespace-only overrides fall through. When env and YAML both provide the public address or SMTP host, env wins in both the gateway and standalone sender. The `From:` header, Message-ID domain, and self-message filter continue to use the public address (`EMAIL_ADDRESS` or its configured fallback); IMAP connection/polling, SMTP connection tests, gateway replies/attachments, and standalone/cron sends use the resolved login. The plugin reads credentials through Hermes's profile-scoped secret reader, so a missing value in a secondary profile never borrows another profile's environment value.

## Verification

Run the repository's canonical test and plugin-validation paths:

```sh
scripts/run_tests.sh tests/plugins/test_email_alias_login.py
hermes plugins validate plugins/email-alias-login
```

Tests use real bundled-plugin discovery and the Email adapter with fake IMAP/SMTP handles. They cover default/env/YAML precedence, inherited-env and profile A → B → A isolation, password masking, required hosts, `From:` retention, and SMTP cleanup after login/send/QUIT errors. No live provider credentials are used.

There is no self-updater or external Python dependency. The subclass preserves bundled Email behavior outside the authentication seams. MIT licensed under the repository license; the bundled Email adapter belongs to Nous Research, and the alias-login plugin changes are by andrexibiza.
