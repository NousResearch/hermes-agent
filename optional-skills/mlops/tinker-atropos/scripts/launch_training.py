import argparse
import asyncio
import sys

from tinker_atropos.config import TinkerAtroposConfig
import tinker_atropos.trainer as trainer_module
from tinker_atropos.trainer import TinkerAtroposTrainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Tinker-Atropos training with YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/default.yaml)",
    )

    # Allow overriding specific config parameters via CLI
    parser.add_argument("--base-model", type=str, help="Override base model")
    parser.add_argument("--lora-rank", type=int, help="Override LoRA rank")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--num-steps", type=int, help="Override number of training steps")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--group-size", type=int, help="Override group size")
    parser.add_argument("--wandb-project", type=str, help="Override wandb project name")
    parser.add_argument("--wandb-group", type=str, help="Override wandb group name")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    return parser.parse_args()


def load_config(args) -> TinkerAtroposConfig:
    """
    Load configuration from YAML file and apply CLI overrides.
    """

    if args.config:
        print(f"Loading config from: {args.config}")
        config = TinkerAtroposConfig.from_yaml(args.config)
    else:
        print("Using default configuration")
        config = TinkerAtroposConfig()

    # Apply CLI overrides
    overrides = {}
    if args.base_model:
        overrides["base_model"] = args.base_model
    if args.lora_rank is not None:
        overrides["lora_rank"] = args.lora_rank
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.num_steps is not None:
        overrides["num_steps"] = args.num_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.group_size is not None:
        overrides["group_size"] = args.group_size
    if args.wandb_project:
        overrides["wandb_project"] = args.wandb_project
    if args.wandb_group:
        overrides["wandb_group"] = args.wandb_group
    if args.no_wandb:
        overrides["use_wandb"] = False

    # Create new config with overrides if any were provided
    if overrides:
        print(f"Applying CLI overrides: {overrides}")
        config_dict = config.to_dict()
        config_dict.update(overrides)
        config = TinkerAtroposConfig(**config_dict)

    return config


async def main():
    """Main entry point for launching training."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Display configuration
    print("\n" + "=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(f"Base Model: {config.base_model}")
    print(f"LoRA Rank: {config.lora_rank}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Training Steps: {config.num_steps}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Group Size: {config.group_size}")
    print(f"Max Token Length (Env): {config.max_token_env_length}")
    print(f"Max Token Length (Trainer): {config.max_token_trainer_length}")
    print(f"Max Workers: {config.max_num_workers}")
    print(f"Wandb Enabled: {config.use_wandb}")
    if config.use_wandb:
        print(f"  Project: {config.wandb_project}")
        print(f"  Group: {config.wandb_group or '(auto-generated)'}")
        print(f"  Run Name: {config.wandb_run_name}-{config.wandb_run_suffix}")
    print(f"Atropos API URL: {config.atropos_api_url}")
    print(f"Inference API URL: {config.inference_api_url}")
    print("=" * 80 + "\n")

    # Initialize trainer
    print("Initializing Tinker-Atropos Trainer...")
    trainer = TinkerAtroposTrainer(config=config)

    # Set the global trainer variable so FastAPI endpoints can access it
    trainer_module.trainer = trainer

    # Start FastAPI inference server in background thread
    from tinker_atropos.trainer import run_fastapi_server
    import threading

    print("Starting FastAPI inference server on port 8001...")
    server_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    server_thread.start()

    print("Waiting for FastAPI server to start...")
    await asyncio.sleep(3)

    # Run training
    try:
        await trainer.run()
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
