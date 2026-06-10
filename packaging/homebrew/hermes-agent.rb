class HermesAgent < Formula
  include Language::Python::Virtualenv

  desc "Self-improving AI agent that creates skills from experience"
  homepage "https://hermes-agent.nousresearch.com"
  url "https://github.com/NousResearch/hermes-agent/archive/refs/tags/v2026.5.16.tar.gz"
  sha256 "c0a554050a50ee9a62f3fa5cd288a167ba5640c42d647d100cdea084b7294143"
  license "MIT"

  depends_on "certifi" => :no_linkage
  depends_on "cryptography" => :no_linkage
  depends_on "libyaml"
  depends_on "python@3.14"

  pypi_packages ignore_packages: %w[certifi cryptography pydantic]

  # Refresh resource stanzas after bumping the source url/version:
  #   brew update-python-resources --print-only hermes-agent

  def install
    venv = virtualenv_create(libexec, "python3.14")
    venv.pip_install resources
    venv.pip_install buildpath

    pkgshare.install "skills", "optional-skills", "ui-tui"

    %w[hermes hermes-agent hermes-acp].each do |exe|
      next unless (libexec/"bin"/exe).exist?

      (bin/exe).write_env_script(
        libexec/"bin"/exe,
        HERMES_BUNDLED_SKILLS: pkgshare/"skills",
        HERMES_OPTIONAL_SKILLS: pkgshare/"optional-skills",
        HERMES_TUI_DIR: pkgshare/"ui-tui",
        HERMES_MANAGED: "homebrew"
      )
    end
  end

  test do
    assert_match "Hermes Agent v#{version}", shell_output("#{bin}/hermes version")

    managed = shell_output("#{bin}/hermes update 2>&1")
    assert_match "managed by Homebrew", managed
    assert_match "brew upgrade hermes-agent", managed
  end
end
