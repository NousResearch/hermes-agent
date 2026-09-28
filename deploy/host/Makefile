# Matter-host tooling. Run from the repo root: make -C deploy/host <target>
REPO := $(abspath ../..)
PY ?= $(REPO)/.venv/bin/python

.PHONY: test dry-run smoke

test:            ## unit tests for the host tooling (no cloud, no Docker)
	cd $(REPO) && $(PY) -m pytest tests/host -q

dry-run:         ## print the image build plan; creates nothing
	./build-image.sh --version $${VERSION:-v0.0.0-dev} --dry-run

smoke:           ## local container smoke test (needs a Docker daemon)
	./smoke.sh
