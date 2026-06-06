# Next.js DB-dependent smoke checks

Use this reference when an All To One session verifies a Next.js app whose login or protected pages depend on Postgres/Neon.

## Durable lesson

Do not treat `lint`/`build` success as full runtime verification. A Next.js app can compile and serve public pages while login, dashboard, tickets, or protected routes hang because Server Actions or data loaders are waiting on a database connection.

## Minimal sequence

1. Run code-layer checks first:
   ```bash
   npm run lint
   npm run build
   ```
2. If smoke/login touches DB, run a minimal DB probe before smoke:
   ```bash
   node --env-file=.env.local -e "const postgres=require('postgres'); const sql=postgres(process.env.POSTGRES_URL||process.env.DATABASE_URL,{max:1,connect_timeout:8}); sql\`select 1 as ok\`.then(r=>{console.log(r[0]); return sql.end()}).catch(e=>{console.error(e.message); process.exit(1)})"
   ```
3. Only after DB probe succeeds, run server + smoke:
   ```bash
   npm run start
   SMOKE_BASE_URL=http://127.0.0.1:3000 npm run smoke
   ```

## How to classify evidence

- `npm run lint && npm run build` exit 0: code/build layer verified.
- Public `GET /login` or `GET /` returns 200: public rendering verified only.
- Login POST hangs/timeouts: likely runtime dependency path blocked; inspect Server Action and DB probe before editing UI.
- DB env vars exist but `select 1` times out: env presence verified, DB connectivity blocked/unverified.

## All To One wording pattern

Write the conclusion as two layers:

> Code layer passed lint/build. Runtime business layer is blocked by DB connectivity, so smoke/login is not verified.

Never write "system verified" when smoke depends on a database that did not connect.
