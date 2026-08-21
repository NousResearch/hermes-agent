# Weixin multi-account shared sessions

Hermes normally keeps Weixin direct-message sessions isolated per sender. Tencent
iLink commonly exposes one QR-created Bot identity to only one personal WeChat
account, so sharing one iLink between two people is not a reliable design.

Use two independently QR-paired iLink accounts behind one Hermes gateway:

```yaml
platforms:
  weixin:
    enabled: true
    extra:
      accounts:
        - name: huihui
          account_id: HUIHUI_ILINK_ACCOUNT_ID
          shared_dm_session: boge-huihui
          shared_dm_users:
            - HUIHUI_WEIXIN_USER_ID
        - name: boge
          account_id: BOGE_ILINK_ACCOUNT_ID
          shared_dm_session: boge-huihui
          shared_dm_users:
            - BOGE_WEIXIN_USER_ID
```

Each account's token is loaded from its QR-login file:

```text
~/.hermes/weixin/accounts/<account_id>.json
```

Do not put tokens in `config.yaml`. The two child adapters keep independent
long-poll connections, credentials, media/context-token stores, and outbound
transports. Their inbound events use the same `shared_dm_session` value, so the
Hermes agent sees one conversation while the response is sent through the child
account that received the triggering message.

Only exact IDs in `shared_dm_users` participate in the shared session. Other
Weixin users remain isolated or are rejected by the normal DM policy. A shared
session intentionally exposes both participants' messages to the same model;
do not add users casually.

The second account is paired without replacing the first one by running the QR
login helper for the same Hermes home and then adding its returned account ID to
the `accounts` list. The QR scan creates a second iLink identity; it is not a
second login to the first identity.
