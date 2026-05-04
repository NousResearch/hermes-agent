# Third-Party Notices

## Google Research CaMeL References

This repository is inspired by the following public research:

- Paper: `Defeating Prompt Injections by Design`
  - https://arxiv.org/abs/2503.18813
  - https://arxiv.org/pdf/2503.18813
- Reference repository:
  - https://github.com/google-research/camel-prompt-injection

## Relationship To This Repository

`hermes-agent-camel` is a Hermes-native implementation of trust-boundary ideas described in the CaMeL paper. It adapts those ideas to Hermes' runtime, tool loop, and skill architecture.

This repository does not claim to be a line-by-line derivative of Google's research code, nor does it claim to reproduce the original evaluation stack exactly.

The public implementation here differs in scope and purpose:

- Google's CaMeL repo is a research artifact intended to reproduce the paper's evaluation setup
- this repository is a runtime hardening integration for Hermes
- the benchmark and validation process here are Hermes-specific

To the best of this repository's authorship intent, Google source code is not vendored into this repository unless explicitly noted in future changes.
