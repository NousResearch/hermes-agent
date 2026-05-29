import pytest
from tinker_atropos.trainer import TinkerAtroposTrainer
from tinker_atropos.config import TinkerAtroposConfig


@pytest.fixture
def trainer():
    config = TinkerAtroposConfig(base_model="meta-llama/Llama-3.1-8B-Instruct")
    trainer = TinkerAtroposTrainer(config=config)
    return trainer


class TestLogprobAlignment:
    def test_basic_alignment_lengths(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [[1, 2, 3, 4, 5, 6, 7, 8]],
                    "inference_logprobs": [[1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.6, 0.5]],
                    "scores": [2.5],
                }
            ]
        }

        datums, group_rewards = trainer.pad_data_to_good_offset(batch_data)
        assert len(datums) == 1

        datum = datums[0]
        input_tokens = datum.model_input.to_ints()
        target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
        logprobs = datum.loss_fn_inputs["logprobs"].to_torch().tolist()
        advantages = datum.loss_fn_inputs["advantages"].to_torch().tolist()

        assert len(input_tokens) == len(target_tokens) == len(logprobs) == len(advantages) == 7

    def test_advantage_masking_for_prompt_tokens(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [1, 2, 3, 4, 5, 6],
                        [1, 2, 3, 7, 8, 9],
                    ],
                    "inference_logprobs": [
                        [1.0, 1.0, 1.0, 0.9, 0.8, 0.7],  # 3 prompt, 3 generated
                        [1.0, 1.0, 1.0, 0.85, 0.75, 0.65],
                    ],
                    "scores": [5.0, 1.0],
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)
        advantages = datums[0].loss_fn_inputs["advantages"].to_torch().tolist()

        assert advantages[0] == 0.0
        assert advantages[1] == 0.0
        assert advantages[2] == 2.0
        assert advantages[3] == 2.0
        assert advantages[4] == 2.0

    def test_all_prompt_tokens(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [[1, 2, 3, 4]],
                    "inference_logprobs": [[1.0, 1.0, 1.0, 1.0]],
                    "scores": [5.0],
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)
        advantages = datums[0].loss_fn_inputs["advantages"].to_torch().tolist()

        assert all(a == 0.0 for a in advantages)
        assert len(advantages) == 3

    def test_all_generated_tokens(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                    ],
                    "inference_logprobs": [
                        [0.9, 0.8, 0.7, 0.6, 0.5],
                        [0.85, 0.75, 0.65, 0.55, 0.45],
                    ],
                    "scores": [6.0, 2.0],
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)
        advantages = datums[0].loss_fn_inputs["advantages"].to_torch().tolist()

        assert all(a == 2.0 for a in advantages)
        assert len(advantages) == 4

    def test_logprob_shift_alignment(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [10, 20, 30, 40, 50],
                        [10, 20, 30, 40, 51],
                    ],
                    "inference_logprobs": [
                        [1.0, 0.9, 0.8, 0.7, 0.6],
                        [1.0, 0.85, 0.75, 0.65, 0.55],
                    ],
                    "scores": [2.0, 1.0],
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)

        target_tokens = datums[0].loss_fn_inputs["target_tokens"].to_torch().tolist()
        logprobs = datums[0].loss_fn_inputs["logprobs"].to_torch().tolist()

        assert target_tokens == [20, 30, 40, 50]
        assert logprobs == pytest.approx([0.9, 0.8, 0.7, 0.6], rel=1e-6)

    def test_zero_advantage_override(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [[1, 2, 3, 4]],
                    "inference_logprobs": [[1.0, 1.0, 0.8, 0.7]],
                    "scores": [0.0],  # Zero advantage
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)
        advantages = datums[0].loss_fn_inputs["advantages"].to_torch().tolist()

        assert all(a == 0.0 for a in advantages)

    def test_negative_advantage(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 6],
                    ],
                    "inference_logprobs": [
                        [1.0, 1.0, 0.8, 0.7, 0.6],
                        [1.0, 1.0, 0.85, 0.75, 0.65],
                    ],
                    "scores": [1.0, 4.0],
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)
        advantages = datums[0].loss_fn_inputs["advantages"].to_torch().tolist()

        assert advantages[0] == 0.0
        assert advantages[1] == -1.5
        assert advantages[2] == -1.5
        assert advantages[3] == -1.5

    def test_multiple_trajectories_in_group(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [1, 2, 3, 4],
                        [1, 2, 3, 5],
                    ],
                    "inference_logprobs": [
                        [1.0, 1.0, 0.9, 0.8],
                        [1.0, 1.0, 0.85, 0.75],
                    ],
                    "scores": [5.0, 3.0],
                }
            ]
        }

        datums, group_rewards = trainer.pad_data_to_good_offset(batch_data)

        assert len(datums) == 2
        assert len(group_rewards) == 1
        assert group_rewards[0] == 4.0

        advantages1 = datums[0].loss_fn_inputs["advantages"].to_torch().tolist()
        assert advantages1[0] == 0.0
        assert advantages1[1] == 1.0
        assert advantages1[2] == 1.0

        advantages2 = datums[1].loss_fn_inputs["advantages"].to_torch().tolist()
        assert advantages2[0] == 0.0
        assert advantages2[1] == -1.0
        assert advantages2[2] == -1.0

    def test_override_set_advantage_to_zero(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [1, 2, 3, 4],
                        [1, 2, 3, 5],
                    ],
                    "inference_logprobs": [
                        [1.0, 1.0, 0.9, 0.8],
                        [1.0, 1.0, 0.85, 0.75],
                    ],
                    "scores": [5.0, 3.0],
                    "overrides": [
                        {},
                        {"set_advantage_to_zero": True},
                    ],
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)

        advantages1 = datums[0].loss_fn_inputs["advantages"].to_torch().tolist()
        assert advantages1[1] == 1.0

        advantages2 = datums[1].loss_fn_inputs["advantages"].to_torch().tolist()
        assert all(a == 0.0 for a in advantages2)

    def test_skip_groups_with_all_zero_advantages(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [1, 2, 3],
                        [1, 2, 4],
                    ],
                    "inference_logprobs": [
                        [1.0, 0.9, 0.8],
                        [1.0, 0.85, 0.75],
                    ],
                    "scores": [3.0, 3.0],
                }
            ]
        }

        datums, group_rewards = trainer.pad_data_to_good_offset(batch_data)

        assert len(datums) == 0
        assert len(group_rewards) == 0


class TestLogprobStatistics:
    def test_logprob_stats_populated(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [[1, 2, 3, 4, 5, 6]],
                    "inference_logprobs": [[1.0, 1.0, 0.9, 0.8, 0.7, 0.6]],
                    "scores": [2.0],
                }
            ]
        }

        trainer.pad_data_to_good_offset(batch_data)

        assert hasattr(trainer, "logprob_stats")
        assert "logprobs/mean" in trainer.logprob_stats
        assert "logprobs/std" in trainer.logprob_stats
        assert "logprobs/min" in trainer.logprob_stats
        assert "logprobs/p50" in trainer.logprob_stats

    def test_logprob_stats_filters_placeholders(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [[1, 2, 3, 4, 5]],
                    "inference_logprobs": [[1.0, 1.0, 0.8, 0.7, 0.6]],
                    "scores": [1.0],
                }
            ]
        }

        trainer.pad_data_to_good_offset(batch_data)

        assert trainer.logprob_stats["logprobs/min"] == pytest.approx(0.6)
        assert 0.65 <= trainer.logprob_stats["logprobs/mean"] <= 0.75

    def test_advantage_stats_populated(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [1, 2, 3],
                        [1, 2, 4],
                        [1, 2, 5],
                    ],
                    "inference_logprobs": [
                        [1.0, 0.9, 0.8],
                        [1.0, 0.85, 0.75],
                        [1.0, 0.8, 0.7],
                    ],
                    "scores": [5.0, 3.0, 2.0],
                }
            ]
        }

        trainer.pad_data_to_good_offset(batch_data)

        assert hasattr(trainer, "advantage_stats")
        assert "advantages/mean" in trainer.advantage_stats
        assert "advantages/std" in trainer.advantage_stats
        assert "advantages/sum" in trainer.advantage_stats


class TestEdgeCases:
    def test_minimal_two_token_sequence(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [1, 2],
                        [1, 3],
                    ],
                    "inference_logprobs": [
                        [1.0, 0.9],
                        [1.0, 0.8],
                    ],
                    "scores": [2.0, 1.0],
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)

        assert len(datums) == 2
        input_tokens = datums[0].model_input.to_ints()
        assert len(input_tokens) == 1

    def test_logprob_boundary_value(self, trainer):
        batch_data = {
            "batch": [
                {
                    "tokens": [
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 6],
                    ],
                    "inference_logprobs": [
                        [1.0, 1.0, 0.99, 0.8, 0.7],
                        [1.0, 1.0, 0.95, 0.75, 0.65],
                    ],
                    "scores": [3.0, 1.0],
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)
        advantages = datums[0].loss_fn_inputs["advantages"].to_torch().tolist()

        assert advantages[0] == 0.0
        assert advantages[1] == 1.0

    def test_empty_batch(self, trainer):
        batch_data = {"batch": []}

        datums, group_rewards = trainer.pad_data_to_good_offset(batch_data)

        assert len(datums) == 0
        assert len(group_rewards) == 0

    def test_very_long_sequence(self, trainer):
        n = 100
        tokens1 = list(range(n))
        tokens2 = list(range(1000, 1000 + n))
        logprobs1 = [1.0] * 10 + [0.8] * (n - 10)
        logprobs2 = [1.0] * 10 + [0.7] * (n - 10)

        batch_data = {
            "batch": [
                {
                    "tokens": [tokens1, tokens2],
                    "inference_logprobs": [logprobs1, logprobs2],
                    "scores": [2.5, 1.5],
                }
            ]
        }

        datums, _ = trainer.pad_data_to_good_offset(batch_data)

        assert len(datums) == 2
        advantages = datums[0].loss_fn_inputs["advantages"].to_torch().tolist()

        masked_count = sum(1 for a in advantages if a == 0.0)
        assert masked_count == 9
