import asyncio
from typing import Any

from pydantic import BaseModel
from swerex.deployment.docker import DockerDeployment
from swerex.runtime.abstract import Command as RexCommand

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class SwerexDockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    deployment_extra_kwargs: dict[str, Any] = {}
    """Extra kwargs to pass to DockerDeployment."""


class SwerexDockerEnvironment:
    def __init__(self, **kwargs):
        """This class executes bash commands in a Docker container using SWE-ReX for sandboxing."""
        self.config = SwerexDockerEnvironmentConfig(**kwargs)
        self.deployment = DockerDeployment(image=self.config.image, **self.config.deployment_extra_kwargs)
        asyncio.run(self.deployment.start())

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the environment and return the raw output."""
        command = action.get("command", "")
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
