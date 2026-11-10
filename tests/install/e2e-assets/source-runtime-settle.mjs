import fs from 'node:fs';
import path from 'node:path';

/**
 * Resolve the installation-owned launcher used to finish a source update.
 *
 * Windows PM installs can contain two launchers during an update. A locked
 * executable can remain beside the current command file. Select the command
 * file explicitly because extensionless lookup would choose the executable.
 *
 * @param {string} root
 * @param {NodeJS.ProcessEnv | Record<string, string>} env
 * @param {NodeJS.Platform} platform
 * @returns {{launcher: string, command: string, args: string[], windowsVerbatimArguments: boolean}}
 */
export function sourceRuntimeSettleCommand(root, env, platform = process.platform) {
  const local = path.join(root, '.hermes', 'bin');
  const candidates = platform === 'win32'
    ? [path.join(local, 'hermes.cmd'), path.join(local, 'hermes.exe'),
      path.join(root, 'venv', 'Scripts', 'hermes.exe')]
    : [path.join(local, 'hermes'), path.join(root, 'venv', 'bin', 'hermes')];
  const launcher = candidates.find(candidate => fs.existsSync(candidate));
  if (!launcher) throw new Error(`No source launcher available to settle ${root}`);

  if (platform !== 'win32' || path.extname(launcher).toLowerCase() !== '.cmd') {
    return { launcher, command: launcher, args: ['status'], windowsVerbatimArguments: false };
  }
  void env;
  // PM's fallback command launcher embeds the Python bootstrap in a base64
  // `-c` argument. Running that .cmd through cmd.exe constrains the already
  // long command to 8191 characters. Drive the bootstrap's prepare_launch seam
  // directly: it owns lazy dependency/product completion, while continuing
  // through a redundant CLI command can inherit update children and never exit.
  const commandFile = fs.readFileSync(launcher, 'utf8');
  const generated = commandFile.match(/^\s*@?"([^"\r\n]+)"\s+-I(?:\s|$)/m);
  if (!generated) throw new Error(`Unrecognized source command launcher: ${launcher}`);
  const command = generated[1];
  // `root` can be an install of an older release (the update e2e scripts settle the OLD install before
  // updating it): take the layout that exists, the package layout first.
  const layouts = [
    { file: path.join(root, 'hermes_agent', 'hermes_cli', 'venv_sync.py'), module: 'hermes_agent.hermes_cli.venv_sync' },
    { file: path.join(root, 'hermes_cli', 'venv_sync.py'), module: 'hermes_cli.venv_sync' },
  ];
  const layout = layouts.find(candidate => fs.existsSync(candidate.file));
  if (!fs.existsSync(command)) throw new Error(`Source launcher Python does not exist: ${command}`);
  if (!layout) throw new Error(`Source update preparation does not exist: ${layouts.map(candidate => candidate.file).join(' or ')}`);
  const code = `import pathlib, sys; sys.path.insert(0, ${JSON.stringify(root)}); from ${layout.module} import prepare_launch; prepare_launch(pathlib.Path(${JSON.stringify(root)}), ['status'])`;
  return { launcher, command, args: ['-I', '-B', '-c', code], windowsVerbatimArguments: false };
}
