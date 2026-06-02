#!/bin/bash

setup_env() {
    export PATH="/usr/local/bin:$PATH"
}

run_tests() {
    echo "Running tests..."
}

main() {
    setup_env
    run_tests
}
