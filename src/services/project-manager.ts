import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as path from 'path';
import { Logger } from '@/infrastructure/logger';
import { DatabaseService } from '@/infrastructure/database';

const execAsync = promisify(exec);

export interface ProjectContext {
  projectId: string;
  userId: string;
  projectPath: string;
  name: string;
  packageManager: 'npm' | 'yarn' | 'pnpm';
}

export interface InstallationResult {
  success: boolean;
  packageName: string;
  version: string;
  message: string;
  installedAt: Date;
  error?: string;
  rollbackNeeded?: boolean;
}

export interface ProjectPackage {
  name: string;
  version: string;
  description: string;
  dependencies: Record<string, string>;
  devDependencies: Record<string, string>;
}

export class ProjectManager {
  private readonly logger: Logger;
  private readonly database: DatabaseService;
  private activeProjects: Map<string, ProjectContext> = new Map();

  constructor(database: DatabaseService, logger: Logger) {
    this.database = database;
    this.logger = logger;
  }

  async registerProject(
    userId: string,
    projectPath: string,
    projectName?: string
  ): Promise<ProjectContext> {
    try {
      const normalizedPath = path.resolve(projectPath);
      const packageJsonPath = path.join(normalizedPath, 'package.json');

      const stats = await fs.stat(normalizedPath);
      if (!stats.isDirectory()) {
        throw new Error(`Path is not a directory: ${projectPath}`);
      }

      const packageJson = await fs.readFile(packageJsonPath, 'utf-8');
      const pkg = JSON.parse(packageJson);

      const projectId = `${userId}-${Date.now()}`;
      const packageManager = this.detectPackageManager(normalizedPath);

      const context: ProjectContext = {
        projectId,
        userId,
        projectPath: normalizedPath,
        name: projectName || pkg.name || 'Unnamed Project',
        packageManager,
      };

      this.activeProjects.set(projectId, context);

      const sql = `
        INSERT INTO user_projects (user_id, project_id, project_path, project_name, package_manager)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (user_id, project_id) DO NOTHING
      `;

      await this.database.execute(sql, [
        userId,
        projectId,
        normalizedPath,
        context.name,
        packageManager,
      ]);

      this.logger.info(`Project registered: ${projectId}`, { userId, projectPath });

      return context;
    } catch (error) {
      this.logger.error(`Failed to register project: ${projectPath}`, error as Error);
      throw error;
    }
  }

  private detectPackageManager(projectPath: string): 'npm' | 'yarn' | 'pnpm' {
    const yarnLock = path.join(projectPath, 'yarn.lock');
    const pnpmLock = path.join(projectPath, 'pnpm-lock.yaml');

    if (fs.existsSync(yarnLock)) return 'yarn';
    if (fs.existsSync(pnpmLock)) return 'pnpm';
    return 'npm';
  }

  async getProject(projectId: string): Promise<ProjectContext | null> {
    return this.activeProjects.get(projectId) || null;
  }

  async getUserProjects(userId: string): Promise<ProjectContext[]> {
    try {
      const sql = `
        SELECT project_id, project_path, project_name, package_manager
        FROM user_projects
        WHERE user_id = $1
        ORDER BY created_at DESC
      `;

      const results = await this.database.query(sql, [userId]);

      return results.map((row) => ({
        projectId: row.project_id,
        userId,
        projectPath: row.project_path,
        name: row.project_name,
        packageManager: row.package_manager,
      }));
    } catch (error) {
      this.logger.error('Failed to fetch user projects', error as Error);
      return [];
    }
  }

  async getProjectPackages(projectId: string): Promise<ProjectPackage | null> {
    try {
      const project = await this.getProject(projectId);
      if (!project) return null;

      const packageJsonPath = path.join(project.projectPath, 'package.json');
      const packageJson = await fs.readFile(packageJsonPath, 'utf-8');
      const pkg = JSON.parse(packageJson);

      return {
        name: pkg.name,
        version: pkg.version,
        description: pkg.description,
        dependencies: pkg.dependencies || {},
        devDependencies: pkg.devDependencies || {},
      };
    } catch (error) {
      this.logger.error(`Failed to get project packages for ${projectId}`, error as Error);
      return null;
    }
  }

  async installPackage(
    projectId: string,
    packageName: string,
    version?: string,
    isDev: boolean = false,
    userId?: string
  ): Promise<InstallationResult> {
    const project = await this.getProject(projectId);
    if (!project) {
      throw new Error(`Project not found: ${projectId}`);
    }

    const installationId = `${projectId}-${Date.now()}`;
    const startTime = Date.now();

    try {
      const packageSpec = version ? `${packageName}@${version}` : packageName;
      const saveFlag = isDev ? '--save-dev' : '--save';

      const command = `cd "${project.projectPath}" && ${project.packageManager} install ${packageSpec} ${saveFlag}`;

      this.logger.info(`Installing package: ${packageSpec}`, {
        projectId,
        command,
      });

      const { stdout, stderr } = await execAsync(command, {
        timeout: 300000,
        maxBuffer: 10 * 1024 * 1024,
      });

      this.logger.debug(`NPM install output: ${stdout}`);

      const resolvedVersion = await this.getInstalledVersion(projectId, packageName);

      const result: InstallationResult = {
        success: true,
        packageName,
        version: resolvedVersion || version || 'latest',
        message: `✅ Successfully installed ${packageName}${version ? `@${version}` : ''}`,
        installedAt: new Date(),
      };

      await this.recordInstallation(
        projectId,
        userId,
        packageName,
        resolvedVersion || version || 'latest',
        'success',
        null,
        Date.now() - startTime
      );

      return result;
    } catch (error) {
      const errorMessage = (error as Error).message;
      this.logger.error(`Package installation failed: ${packageName}`, error as Error);

      await this.recordInstallation(
        projectId,
        userId,
        packageName,
        version || 'latest',
        'failed',
        errorMessage,
        Date.now() - startTime
      );

      return {
        success: false,
        packageName,
        version: version || 'latest',
        message: `❌ Installation failed: ${errorMessage}`,
        installedAt: new Date(),
        error: errorMessage,
        rollbackNeeded: true,
      };
    }
  }

  private async getInstalledVersion(projectId: string, packageName: string): Promise<string | null> {
    try {
      const project = await this.getProject(projectId);
      if (!project) return null;

      const packageJsonPath = path.join(project.projectPath, 'package.json');
      const packageJson = await fs.readFile(packageJsonPath, 'utf-8');
      const pkg = JSON.parse(packageJson);

      const version =
        pkg.dependencies?.[packageName] || pkg.devDependencies?.[packageName] || null;

      return version;
    } catch (error) {
      this.logger.error(`Failed to get installed version for ${packageName}`, error as Error);
      return null;
    }
  }

  private async recordInstallation(
    projectId: string,
    userId: string | undefined,
    packageName: string,
    version: string,
    status: 'success' | 'failed',
    errorMessage: string | null,
    duration: number
  ): Promise<void> {
    try {
      const sql = `
        INSERT INTO package_installations 
        (project_id, user_id, package_name, version, status, error_message, duration_ms)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
      `;

      await this.database.execute(sql, [
        projectId,
        userId,
        packageName,
        version,
        status,
        errorMessage,
        duration,
      ]);
    } catch (error) {
      this.logger.error('Failed to record installation', error as Error);
    }
  }

  async uninstallPackage(projectId: string, packageName: string): Promise<boolean> {
    try {
      const project = await this.getProject(projectId);
      if (!project) return false;

      const command = `cd "${project.projectPath}" && ${project.packageManager} uninstall ${packageName}`;

      await execAsync(command, { timeout: 120000 });

      this.logger.info(`Uninstalled package: ${packageName}`, { projectId });
      return true;
    } catch (error) {
      this.logger.error(`Failed to uninstall ${packageName}`, error as Error);
      return false;
    }
  }

  async updateProject(projectId: string): Promise<boolean> {
    try {
      const project = await this.getProject(projectId);
      if (!project) return false;

      const command = `cd "${project.projectPath}" && ${project.packageManager} update`;

      await execAsync(command, { timeout: 300000 });

      this.logger.info(`Updated project dependencies`, { projectId });
      return true;
    } catch (error) {
      this.logger.error(`Failed to update project`, error as Error);
      return false;
    }
  }

  async auditProject(projectId: string): Promise<any> {
    try {
      const project = await this.getProject(projectId);
      if (!project) return null;

      const command = `cd "${project.projectPath}" && npm audit --json`;

      const { stdout } = await execAsync(command, { timeout: 60000 });
      return JSON.parse(stdout);
    } catch (error) {
      this.logger.error(`Audit failed for project`, error as Error);
      return null;
    }
  }
}
