# Nomad Configuration for Singularity/Apptainer Sandbox
# Run with: nomad agent -dev -config=nomad-singularity.hcl
#
# This enables the raw_exec driver, which can be used to run Apptainer
# commands on hosts where Docker is unavailable.
#
# NOTE: Hermes-Agent's Nomad backend support is draft; this file is provided
# as a starting point for local testing.

client {
  enabled = true

  options {
    "driver.raw_exec.enable" = "1"
  }
}

plugin "raw_exec" {
  config {
    enabled = true
  }
}

# If you have a dedicated Nomad Singularity/Apptainer driver plugin installed,
# you can configure that instead of raw_exec.
