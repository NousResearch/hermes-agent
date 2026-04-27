import re
with open("hermes_cli/web_server.py", "r") as f:
    content = f.read()

# Define the models
models = """
class _RunCommandBody(BaseModel):
    command: str
    cwd: Optional[str] = None
    timeout_seconds: int = 120
"""

# Append models after _AddCodeSessionEventBody
content = content.replace(
    "class _AddCodeSessionEventBody(BaseModel):\n    type: str\n    message: Optional[str] = None\n    payload: Optional[dict] = None\n",
    "class _AddCodeSessionEventBody(BaseModel):\n    type: str\n    message: Optional[str] = None\n    payload: Optional[dict] = None\n" + models + "\n"
)

endpoints = """
@app.get("/api/code/sessions/{code_session_id}/commands")
async def list_commands(code_session_id: str):
    from hermes_cli.code.command_runner import CommandRunnerService
    try:
        svc = CommandRunnerService(realtime_hub=_REALTIME_HUB)
        commands = svc.list_commands(code_session_id)
        return {"commands": commands, "total": len(commands)}
    except Exception as exc:
        _log.error("list_commands failed for %s: %s", code_session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/code/sessions/{code_session_id}/commands/run")
async def run_command(code_session_id: str, body: _RunCommandBody):
    from hermes_cli.code.command_runner import CommandRunnerService
    try:
        svc = CommandRunnerService(realtime_hub=_REALTIME_HUB)
        
        # 1. Create the command record
        cmd = svc.create_command(
            code_session_id=code_session_id,
            command=body.command,
            cwd=body.cwd,
            timeout_seconds=body.timeout_seconds
        )
        
        # Emit command.started
        try:
            await _REALTIME_HUB.broadcast(
                "command.started",
                {"payload": {"command": cmd}}
            )
        except Exception:
            pass

        # 2. If it's blocked or needs approval, return it immediately without running
        if cmd["status"] in ("blocked", "needs_approval"):
            return {"command": cmd}

        # 3. Run the command sync
        updated_cmd = svc.run_command_sync(cmd["id"])
        
        # 4. Emit completed/failed/timeout
        try:
            await _REALTIME_HUB.broadcast(
                f"command.{updated_cmd['status']}",
                {"payload": {"command": updated_cmd}}
            )
            # Also emit command.output
            if updated_cmd.get("stdout") or updated_cmd.get("stderr"):
                await _REALTIME_HUB.broadcast(
                    "command.output",
                    {
                        "payload": {
                            "command_id": updated_cmd["id"],
                            "stdout": updated_cmd.get("stdout", ""),
                            "stderr": updated_cmd.get("stderr", "")
                        }
                    }
                )
        except Exception:
            pass
            
        return {"command": updated_cmd}
        
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as exc:
        _log.error("run_command failed for %s: %s", code_session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/code/commands/{command_id}")
async def get_command(command_id: str):
    from hermes_cli.code.command_runner import CommandRunnerService
    try:
        svc = CommandRunnerService(realtime_hub=_REALTIME_HUB)
        cmd = svc.get_command(command_id)
        if not cmd:
            raise HTTPException(status_code=404, detail="Command not found")
        return {"command": cmd}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error("get_command failed for %s: %s", command_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/code/commands/{command_id}/cancel")
async def cancel_command(command_id: str):
    from hermes_cli.code.command_runner import CommandRunnerService
    try:
        svc = CommandRunnerService(realtime_hub=_REALTIME_HUB)
        cmd = svc.get_command(command_id)
        if not cmd:
            raise HTTPException(status_code=404, detail="Command not found")
            
        if cmd["status"] in ("completed", "failed", "timeout", "cancelled"):
            return {
                "command": cmd,
                "message": "Command is not running"
            }
            
        updated_cmd = svc.cancel_command(command_id)
        
        try:
            await _REALTIME_HUB.broadcast(
                "command.cancelled",
                {"payload": {"command": updated_cmd}}
            )
        except Exception:
            pass
            
        return {"command": updated_cmd}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error("cancel_command failed for %s: %s", command_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
"""

# Append endpoints at the end of the file or before the execution blocks
content = content + "\n\n" + endpoints

with open("hermes_cli/web_server.py", "w") as f:
    f.write(content)
