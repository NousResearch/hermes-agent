import { Logger } from '@/infrastructure/logger';
import { ConfigService } from '@/infrastructure/config';
import { ApiClient } from '@/infrastructure/api-client';

export interface NPMPackage {
  name: string;
  version: string;
  description: string;
  keywords: string[];
  homepage: string;
  repository: string;
  author: string;
  license: string;
  downloads: number;
}

export interface SearchResult {
  packages: NPMPackage[];
  total: number;
}

export class NPMPackageManager {
  private npmClient: ApiClient;
  private readonly logger: Logger;
  private readonly config: ConfigService;

  constructor(config: ConfigService, logger: Logger) {
    this.config = config;
    this.logger = logger;
    this.npmClient = new ApiClient(
      {
        baseURL: config.npmRegistryUrl,
        timeout: 15000,
        headers: config.npmToken
          ? { Authorization: `Bearer ${config.npmToken}` }
          : {},
      },
      logger
    );
  }

  async searchPackages(query: string, limit: number = 10): Promise<SearchResult> {
    try {
      const encodedQuery = encodeURIComponent(query);
      const response = await this.npmClient.get<any>(
        `/-/v1/search?text=${encodedQuery}&size=${limit}`
      );

      const packages = response.objects.map((obj: any) => ({
        name: obj.package.name,
        version: obj.package.version,
        description: obj.package.description,
        keywords: obj.package.keywords || [],
        homepage: obj.package.links?.homepage || '',
        repository: obj.package.links?.repository || '',
        author: obj.package.author?.name || 'Unknown',
        license: obj.package.license || 'Unknown',
        downloads: obj.score?.detail?.popularity || 0,
      }));

      this.logger.info(`NPM search completed: ${query}`, {
        resultCount: packages.length,
      });

      return {
        packages,
        total: response.total,
      };
    } catch (error) {
      this.logger.error(`NPM search failed: ${query}`, error as Error);
      throw error;
    }
  }

  async getPackageInfo(packageName: string): Promise<any> {
    try {
      const response = await this.npmClient.get<any>(`/${packageName}`);
      return response;
    } catch (error) {
      this.logger.error(`Failed to get package info: ${packageName}`, error as Error);
      throw error;
    }
  }

  async getPackageVersions(packageName: string): Promise<string[]> {
    try {
      const info = await this.getPackageInfo(packageName);
      return Object.keys(info.versions || {});
    } catch (error) {
      this.logger.error(`Failed to get versions for: ${packageName}`, error as Error);
      throw error;
    }
  }

  async installPackage(
    packageName: string,
    version?: string,
    userId?: string
  ): Promise<{ success: boolean; message: string; installedVersion?: string }> {
    try {
      const packageSpec = version ? `${packageName}@${version}` : packageName;

      this.logger.info(`Validating NPM package: ${packageSpec}`, { userId });

      const isValid = await this.validatePackage(packageName);
      if (!isValid) {
        return {
          success: false,
          message: `Package not found in NPM registry: ${packageName}`,
        };
      }

      return {
        success: true,
        message: `Package validated: ${packageSpec}. Use /install to proceed with installation in your project.`,
        installedVersion: version,
      };
    } catch (error) {
      const errorMessage = `Failed to validate package: ${packageName}`;
      this.logger.error(errorMessage, error as Error);
      return {
        success: false,
        message: errorMessage,
      };
    }
  }

  async getPackageSimilar(packageName: string, limit: number = 5): Promise<NPMPackage[]> {
    try {
      const info = await this.getPackageInfo(packageName);
      const keywords = info.keywords || [];

      if (keywords.length === 0) {
        return [];
      }

      const searchQuery = keywords.slice(0, 3).join(' ');
      const results = await this.searchPackages(searchQuery, limit + 1);

      return results.packages
        .filter((pkg) => pkg.name !== packageName)
        .slice(0, limit);
    } catch (error) {
      this.logger.error(`Failed to find similar packages for: ${packageName}`, error as Error);
      return [];
    }
  }

  async validatePackage(packageName: string): Promise<boolean> {
    try {
      await this.getPackageInfo(packageName);
      return true;
    } catch {
      return false;
    }
  }

  parsePackageString(packageString: string): { name: string; version?: string } {
    const match = packageString.match(/^(@?[^@]+)(?:@(.+))?$/);

    if (!match) {
      throw new Error(`Invalid package string: ${packageString}`);
    }

    return {
      name: match[1],
      version: match[2],
    };
  }
}
