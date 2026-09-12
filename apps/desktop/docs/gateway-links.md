# Open a remote gateway in Hermes Desktop

A website or terminal can open this link in Hermes Desktop:

```text
hermes://gateway/connect?url=https%3A%2F%2Fgateway.example.com%2Fteam%2Falice&name=Work
```

Build the query with `URLSearchParams`, encoding the complete HTTPS gateway
base URL as `url` and an optional display name as `name`. Desktop preserves the
gateway's path prefix. Links containing credentials, a query or fragment in
the gateway URL, other schemes, or unexpected parameters are rejected.

Desktop opens Gateway settings and displays the target address. The user
confirms before Desktop contacts the gateway or saves anything. The existing
gateway sign-in window handles authentication; credentials are never accepted
through a deep link. Canceled sign-in leaves the registry unchanged. A failed
connection leaves the saved entry available for retry and reports the error.

The connection is added to the existing registry. Opening the same URL again
reuses its entry, and other connections and the primary startup choice remain
intact. Switching uses the ordinary authenticated WebSocket connection flow.
The existing Electron URL handler queues cold-start links and also delivers
links to an already-running app.

The calling website should say that it is opening Desktop, provide an install
or update link if no connection prompt appears, and offer the gateway URL for
manual setup. Launching a URI does not confirm that the app is installed or
that the remote connection succeeded. Older Desktop versions do not support
this handoff.

Validation from `apps/desktop`:

```sh
bunx vitest run src/lib/gateway-connect-link.test.ts src/lib/connect-linked-gateway.test.ts src/lib/deeplink-routes.test.ts
```

Before release, exercise both cold-start and already-open handoff, canceled
sign-in, two repeated opens of the same gateway, a failed WebSocket handshake,
and a restart with the saved connection.
