# This workflow uses actions that are not certified by GitHub.
# They are provided by a third-party and are governed by
# separate terms of service, privacy policy, and support
# documentation.

# A sample workflow which sets up Snyk to analyze the full Snyk platform (Snyk Open Source, Snyk Code,
# Snyk Container and Snyk Infrastructure as Code)
# The setup installs the Snyk CLI - for more details on the possible commands
# check https://docs.snyk.io/snyk-cli/cli-reference
# The results of Snyk Code are then uploaded to GitHub Security Code Scanning
#
# In order to use the Snyk Action you will need to have a Snyk API token.
# More details in https://github.com/snyk/actions#getting-your-snyk-token
# or you can signup for free at https://snyk.io/login
#
# For more examples, including how to limit scans to only high-severity issues
# and fail PR checks, see https://github.com/snyk/actions/

name: Snyk Security

on:
  push:
    branches: ["main" ]
  pull_request:
    branches: ["main"]

permissions:
  contents: read

jobs:
  snyk:
    permissions:
      contents: read # for actions/checkout to fetch code
      security-events: write # for github/codeql-action/upload-sarif to upload SARIF results
      actions: read # only required for a private repository by github/codeql-action/upload-sarif to get the Action run status
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Snyk CLI to check for security issues
        # Snyk can be used to break the build when it detects security issues.
        # In this case we want to upload the SAST issues to GitHub Code Scanning
        uses: snyk/actions/setup@806182742461562b67788a64410098c9d9b96adb

        # For Snyk Open Source you must first set up the development environment for your application's dependencies
        # For example for Node
        #- uses: actions/setup-node@v4
        #  with:
        #    node-version: 20

        env:
          # This is where you will need to introduce the Snyk API token created with your Snyk account
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

        # Runs Snyk Code (SAST) analysis and uploads result into GitHub.
        # Use || true to not fail the pipeline
      - name: Snyk Code test
        run: snyk code test --sarif > snyk-code.sarif # || true

        # Runs Snyk Open Source (SCA) analysis and uploads result to Snyk.
      - name: Snyk Open Source monitor
        run: snyk monitor --all-projects

        # Runs Snyk Infrastructure as Code (IaC) analysis and uploads result to Snyk.
        # Use || true to not fail the pipeline.
      - name: Snyk IaC test and report
        run: snyk iac test --report # || true

        # Build the docker image for testing
      - name: Build a Docker image
        run: docker build -t your/image-to-test .
        # Runs Snyk Container (Container and SCA) analysis and uploads result to Snyk.
      - name: Snyk Container monitor
        run: snyk container monitor your/image-to-test --file=Dockerfile

        # Push the Snyk Code results into GitHub Code Scanning tab
      - name: Upload result to GitHub Code Scanning
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: snyk-code.sarif
