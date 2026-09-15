#!/usr/bin/env bash
# Stand-in for ssh inside the dev sandbox.
#
# install.sh and `hermes update` clone over ssh first (git@github.com:...), so
# the sandbox needs an `ssh` that answers. Rather than run a real sshd, this
# ignores the host, user, and command git asked for and speaks the
# upload-pack protocol directly against the sandbox's bare repo -- which is
# what makes the ssh-first code path exercisable with no keys, no known_hosts,
# and no network.
#
# GIT_UPLOAD_PACK is substituted by dev-sandbox.sh when it installs this shim,
# because the host's git-upload-pack is not necessarily on the sandbox PATH.
exec @GIT_UPLOAD_PACK@ /work/repos/hermes-agent.git
