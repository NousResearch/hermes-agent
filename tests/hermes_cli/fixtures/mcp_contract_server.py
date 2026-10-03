"""Small stdio MCP peer for contract CLI integration tests. No business tools run."""

import json
import os
from pathlib import Path
import sys


def main():
    data_path, audit_path = map(Path, sys.argv[1:])
    data = json.loads(data_path.read_text(encoding="utf-8"))
    with audit_path.open("a", encoding="utf-8") as audit:
        audit.write(json.dumps({"event": "start", "pid": os.getpid()}) + "\n")
        audit.flush()
        for line in sys.stdin:
            request = json.loads(line)
            method = request["method"]
            audit.write(json.dumps({"method": method, "params": request.get("params", {})}) + "\n")
            audit.flush()
            if "id" not in request:
                continue
            response = {"jsonrpc": "2.0", "id": request["id"]}
            if method == "initialize":
                response["result"] = {"protocolVersion": request["params"]["protocolVersion"],
                                      "capabilities": {"tools": {}},
                                      "serverInfo": {"name": "contract-fixture", "version": "1.0"}}
            elif method == "tools/list":
                cursor = request.get("params", {}).get("cursor")
                if data.get("mode") == "error" and cursor:
                    response["error"] = {"code": -32603, "message": "Authorization: Bearer SYNTHETIC_FIXTURE_TOKEN"}
                else:
                    index = int(cursor or 0)
                    response["result"] = {"tools": data["tools"][index:index + 1]}
                    if data.get("mode") == "cycle":
                        response["result"]["nextCursor"] = "0"
                    elif index + 1 < len(data["tools"]):
                        response["result"]["nextCursor"] = str(index + 1)
            else:
                response["error"] = {"code": -32601, "message": "Unexpected method in snapshot probe"}
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
