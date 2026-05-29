import random
import string
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml
from pydantic import BaseModel, Field


def generate_run_suffix() -> str:
    """Generate a random 4-character suffix for unique wandb run names."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))


class EnvConfig(BaseModel):
    """Environment configuration - matches Atropos BaseEnvConfig"""

    group_size: int = 16
    batch_size: int = 128
    max_batches_offpolicy: int = 3
    tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    use_wandb: bool = True
    rollout_server_url: str = "http://localhost:8000"
    wandb_name: str = "atropos-tinker-env"
    ensure_scores_are_not_same: bool = False
    max_token_length: int = 256
    max_num_workers: int = 24
    total_steps: int = 50
    steps_per_eval: int = 100


class OpenAIServerConfig(BaseModel):
    """OpenAI-compatible server configuration"""

    model_name: str
    base_url: str
    api_key: str = "x"
    weight: float = 1.0
    num_requests_for_eval: int = 256


class TinkerConfig(BaseModel):
    """Tinker-specific configuration for LoRA training"""

    lora_rank: int = 32
    learning_rate: float = 4e-5
    max_token_trainer_length: int = 2048
    checkpoint_dir: str = "./temp/"
    save_checkpoint_interval: int = 0

    # Wandb configuration for trainer
    wandb_project: str = "atropos-tinker"
    wandb_group: Optional[str] = None
    wandb_run_name: str = "atropos-tinker-run"
    wandb_run_suffix: str = Field(default_factory=generate_run_suffix)


class TinkerAtroposConfig(BaseModel):
    """Complete Tinker-Atropos configuration"""

    env: EnvConfig = Field(default_factory=EnvConfig)
    openai: List[OpenAIServerConfig] = Field(
        default_factory=lambda: [
            OpenAIServerConfig(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                base_url="http://localhost:8001/v1",
            )
        ]
    )
    tinker: TinkerConfig = Field(default_factory=TinkerConfig)
    slurm: bool = False
    testing: bool = False

    # Convenience properties
    @property
    def base_model(self) -> str:
        """Convenience accessor for model name"""
        return self.env.tokenizer_name

    @property
    def atropos_api_url(self) -> str:
        """Convenience accessor for Atropos API URL"""
        return self.env.rollout_server_url

    @property
    def inference_api_url(self) -> str:
        """Convenience accessor for inference server URL (first OpenAI server)"""
        if self.openai:
            # Remove /v1 suffix if present
            url = self.openai[0].base_url
            return url.rstrip("/v1").rstrip("/")
        return "http://localhost:8001"

    @property
    def group_size(self) -> int:
        return self.env.group_size

    @property
    def batch_size(self) -> int:
        return self.env.batch_size

    @property
    def max_batches_offpolicy(self) -> int:
        return self.env.max_batches_offpolicy

    @property
    def use_wandb(self) -> bool:
        return self.env.use_wandb

    @property
    def num_steps(self) -> int:
        return self.env.total_steps

    @property
    def steps_per_eval(self) -> int:
        return self.env.steps_per_eval

    @property
    def max_token_env_length(self) -> int:
        return self.env.max_token_length

    @property
    def max_num_workers(self) -> int:
        return self.env.max_num_workers

    @property
    def ensure_scores_are_not_same(self) -> bool:
        return self.env.ensure_scores_are_not_same

    @property
    def wandb_run_name(self) -> str:
        return self.tinker.wandb_run_name

    @property
    def wandb_project(self) -> str:
        return self.tinker.wandb_project

    @property
    def wandb_group(self) -> Optional[str]:
        return self.tinker.wandb_group

    @property
    def wandb_run_suffix(self) -> str:
        return self.tinker.wandb_run_suffix

    @property
    def lora_rank(self) -> int:
        return self.tinker.lora_rank

    @property
    def learning_rate(self) -> float:
        return self.tinker.learning_rate

    @property
    def max_token_trainer_length(self) -> int:
        return self.tinker.max_token_trainer_length

    @property
    def checkpoint_dir(self) -> str:
        return self.tinker.checkpoint_dir

    @property
    def save_checkpoint_interval(self) -> int:
        return self.tinker.save_checkpoint_interval

    @property
    def num_requests_for_eval(self) -> int:
        if self.openai:
            return self.openai[0].num_requests_for_eval
        return 256

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TinkerAtroposConfig":
        """
        Load configuration from a YAML file.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        return cls(**yaml_data)
