# Nomad Configuration for Singularity/Apptainer Sandbox
# Run with: nomad agent -dev -config=nomad-singularity.hcl
#
# This uses the raw_exec driver to run Apptainer containers.
# Suitable for HPC environments where Docker cannot run without sudo.

client {
  enabled = true

  options {
    # Enable raw_exec driver for Singularity/Apptainer
    "driver.raw_exec.enable" = "1"
  }
}

# raw_exec driver plugin configuration
plugin "raw_exec" {
  config {
    enabled = true
  }
}

# Optional: If you have the nomad-driver-singularity plugin installed,
# uncomment the following instead of using raw_exec:
# plugin "singularity" {
#   config {
#     enabled = true
#     # Allow bind mounts
#     bind_paths = ["/tmp", "/var/tmp"]
#   }
# }
