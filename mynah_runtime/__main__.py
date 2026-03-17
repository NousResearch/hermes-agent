from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("MYNAH_RUNTIME_HOST", "0.0.0.0")
    port = int(os.getenv("MYNAH_RUNTIME_PORT", "8091"))
    uvicorn.run("mynah_runtime.service:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
