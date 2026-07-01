# Atomic Mail — Auth flow

1. `POST /api/v1/challenge` to get challenge JWT from `Authorization` header.
2. Solve PoW with scrypt (`N=16384,r=8,p=1,dklen=64`).
3. `POST /api/v1/session` with challenge JWT + PoW fields in JSON body.
4. `POST /api/v1/capability` with session JWT to get capability JWT.

JWTs are refreshed automatically and persisted to disk.
