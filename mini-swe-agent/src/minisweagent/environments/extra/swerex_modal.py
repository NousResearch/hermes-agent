import asyncio
from typing import Any

from pydantic import BaseModel
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import Command as RexCommand

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class SwerexModalEnvironmentConfig(BaseModel):
    image: str
    """Image to use for the deployment. Can be:
    - Dockerhub image name (e.g. `python:3.11-slim`)
    - ECR image name (e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:tag`)
    - Path to a Dockerfile
    """
    cwd: str = "/"
    """Working directory in which to execute commands."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    env: dict[str, str] = {}
    """Environment variables to set when executing commands."""
    startup_timeout: float = 60.0
    """The time to wait for the runtime to start."""
    runtime_timeout: float = 3600.0
    """The runtime timeout (how long the Modal sandbox can stay alive)."""
    deployment_timeout: float = 3600.0
    """The deployment timeout."""
    install_pipx: bool = True
    """Whether to install pipx in the container (required for swe-rex runtime)."""
    modal_sandbox_kwargs: dict[str, Any] = {}
    """Additional arguments to pass to `modal.Sandbox.create`."""


class SwerexModalEnvironment:
    def __init__(self, **kwargs):
        """This class executes bash commands in a Modal sandbox using SWE-ReX for remote execution.

        Modal (https://modal.com) provides serverless cloud compute that can be used to run
        sandboxed environments. This environment class uses SWE-ReX's ModalDeployment to
        create and manage Modal sandboxes for command execution.

        This is useful for:
        - Training coding agents at scale with remote execution
        - Running evaluations in isolated cloud environments
        - Parallel execution across many instances

        See `SwerexModalEnvironmentConfig` for keyword arguments.
        """
        self.config = SwerexModalEnvironmentConfig(**kwargs)
        self.deployment = ModalDeployment(
            image=self.config.image,
            startup_timeout=self.config.startup_timeout,
            runtime_timeout=self.config.runtime_timeout,
            deployment_timeout=self.config.deployment_timeout,
            install_pipx=self.config.install_pipx,
            modal_sandbox_kwargs=self.config.modal_sandbox_kwargs,
        )
        asyncio.run(self.deployment.start())

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the environment and return the raw output."""
        command = action.get("command", "") if isinstance(action, dict) else action
        try:
            result = asyncio.run(
                self.deployment.runtime.execute(
                    RexCommand(
                        command=command,
                        shell=True,
                        check=False,
                        cwd=cwd or self.config.cwd,
                        timeout=timeout or self.config.timeout,
                        merge_output_streams=True,
                        env=self.config.env if self.config.env else None,
                    )
                )
            )
            output = {"output": result.stdout, "returncode": result.exit_code, "exception_info": ""}
        except Exception as e:
            output = {
                "output": str(e) if str(e) else "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def stop(self):
        async def _stop():
            await asyncio.wait_for(self.deployment.stop(), timeout=10)

        try:
            asyncio.run(_stop())
        except Exception:
            pass
