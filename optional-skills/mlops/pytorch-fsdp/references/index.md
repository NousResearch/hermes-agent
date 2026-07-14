# PyTorch FSDP Reference Index

`other.md` is the original generated documentation archive shipped with this
skill. It is intentionally kept behind progressive disclosure because it is a
large snapshot and may lag current PyTorch APIs.

Use `search_files` within `other.md` for the exact heading below, then use
`read_file` on only the relevant section.

## FSDP APIs

- `torch.distributed.fsdp.fully_shard` - FSDP2 composable sharding.
- `FullyShardedDataParallel` - FSDP1 wrapper, policies, and state dictionaries.

## Checkpointing and tensors

- `Distributed Checkpoint - torch.distributed.checkpoint`
- `torch.distributed.tensor`

## Process groups and launch

- `Distributed communication package - torch.distributed`
- `Generic Join Context Manager`
- `Torch Distributed Elastic`

## Related parallelism

- `DistributedDataParallel`
- `DDP Communication Hooks`
- `Pipeline Parallelism`
- `Tensor Parallelism - torch.distributed.tensor.parallel`

## Currency rule

The archive records documentation captured in 2025. Use it for concepts and
search terms, but verify signatures and recommendations against the current
official documentation:

- FSDP2: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- FSDP1: https://docs.pytorch.org/docs/stable/fsdp.html
- Distributed checkpointing:
  https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
