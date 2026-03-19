"""
cosmos AWS Manager

"Good news, everyone! I can manage your AWS infrastructure!"

Comprehensive AWS management for sysadmins.

Capabilities:
- EC2 instance management
- IAM user and role management
- S3 bucket operations
- VPC and networking
- CloudWatch monitoring
- Cost exploration
- Security Hub integration
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from loguru import logger


class AWSAuthMethod(Enum):
    """AWS authentication methods."""
    ACCESS_KEY = "access_key"
    PROFILE = "profile"
    INSTANCE_ROLE = "instance_role"
    SSO = "sso"


class EC2State(Enum):
    """EC2 instance states."""
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    SHUTTING_DOWN = "shutting-down"
    TERMINATED = "terminated"


@dataclass
class AWSConfig:
    """AWS configuration."""
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: str = ""  # For temporary credentials
    region: str = "us-east-1"
    profile: str = ""  # AWS profile name
    auth_method: AWSAuthMethod = AWSAuthMethod.ACCESS_KEY


@dataclass
class EC2Instance:
    """EC2 instance representation."""
    instance_id: str
    name: str = ""
    instance_type: str = ""
    state: EC2State = EC2State.PENDING
    public_ip: str = ""
    private_ip: str = ""
    vpc_id: str = ""
    subnet_id: str = ""
    security_groups: List[str] = field(default_factory=list)
    launch_time: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class IAMUser:
    """IAM user representation."""
    user_name: str
    user_id: str = ""
    arn: str = ""
    create_date: Optional[datetime] = None
    password_last_used: Optional[datetime] = None
    path: str = "/"
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class S3Bucket:
    """S3 bucket representation."""
    name: str
    creation_date: Optional[datetime] = None
    region: str = ""


class AWSManager:
    """
    Comprehensive AWS management.

    Prerequisites:
    1. AWS account
    2. IAM user or role with appropriate permissions
    3. Access keys or configured AWS CLI profile

    Uses boto3 when available, falls back to direct API calls.
    """

    def __init__(self, config: Optional[AWSConfig] = None):
        """Initialize AWS manager."""
        self.config = config or AWSConfig()
        self._session = None
        self._clients: Dict[str, Any] = {}

    def _get_session(self):
        """Get or create boto3 session."""
        if self._session is None:
            try:
                import boto3

                if self.config.auth_method == AWSAuthMethod.PROFILE:
                    self._session = boto3.Session(
                        profile_name=self.config.profile,
                        region_name=self.config.region,
                    )
                elif self.config.auth_method == AWSAuthMethod.ACCESS_KEY:
                    self._session = boto3.Session(
                        aws_access_key_id=self.config.access_key_id,
                        aws_secret_access_key=self.config.secret_access_key,
                        aws_session_token=self.config.session_token or None,
                        region_name=self.config.region,
                    )
                else:
                    # Default session (uses environment/instance role)
                    self._session = boto3.Session(region_name=self.config.region)

            except ImportError:
                raise ImportError("boto3 required: pip install boto3")

        return self._session

    def _get_client(self, service: str):
        """Get boto3 client for service."""
        if service not in self._clients:
            session = self._get_session()
            self._clients[service] = session.client(service)
        return self._clients[service]

    def authenticate(self) -> bool:
        """Test authentication by calling STS."""
        try:
            sts = self._get_client("sts")
            identity = sts.get_caller_identity()
            logger.info(f"AWS authenticated as: {identity.get('Arn')}")
            return True
        except Exception as e:
            logger.error(f"AWS authentication failed: {e}")
            return False

    # ========== EC2 Management ==========

    def list_instances(
        self,
        filters: List[Dict] = None,
        instance_ids: List[str] = None,
    ) -> List[EC2Instance]:
        """List EC2 instances."""
        ec2 = self._get_client("ec2")

        params = {}
        if filters:
            params["Filters"] = filters
        if instance_ids:
            params["InstanceIds"] = instance_ids

        response = ec2.describe_instances(**params)

        instances = []
        for reservation in response.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                # Get name from tags
                name = ""
                tags = {}
                for tag in inst.get("Tags", []):
                    tags[tag["Key"]] = tag["Value"]
                    if tag["Key"] == "Name":
                        name = tag["Value"]

                instances.append(EC2Instance(
                    instance_id=inst.get("InstanceId", ""),
                    name=name,
                    instance_type=inst.get("InstanceType", ""),
                    state=EC2State(inst.get("State", {}).get("Name", "pending")),
                    public_ip=inst.get("PublicIpAddress", ""),
                    private_ip=inst.get("PrivateIpAddress", ""),
                    vpc_id=inst.get("VpcId", ""),
                    subnet_id=inst.get("SubnetId", ""),
                    security_groups=[sg["GroupId"] for sg in inst.get("SecurityGroups", [])],
                    launch_time=inst.get("LaunchTime"),
                    tags=tags,
                ))

        return instances

    def start_instances(self, instance_ids: List[str]) -> Dict[str, Any]:
        """Start EC2 instances."""
        ec2 = self._get_client("ec2")
        response = ec2.start_instances(InstanceIds=instance_ids)
        logger.info(f"Started instances: {instance_ids}")
        return response

    def stop_instances(
        self,
        instance_ids: List[str],
        force: bool = False,
    ) -> Dict[str, Any]:
        """Stop EC2 instances."""
        ec2 = self._get_client("ec2")
        response = ec2.stop_instances(InstanceIds=instance_ids, Force=force)
        logger.info(f"Stopped instances: {instance_ids}")
        return response

    def reboot_instances(self, instance_ids: List[str]) -> Dict[str, Any]:
        """Reboot EC2 instances."""
        ec2 = self._get_client("ec2")
        response = ec2.reboot_instances(InstanceIds=instance_ids)
        logger.info(f"Rebooted instances: {instance_ids}")
        return response

    def terminate_instances(self, instance_ids: List[str]) -> Dict[str, Any]:
        """Terminate EC2 instances."""
        ec2 = self._get_client("ec2")
        response = ec2.terminate_instances(InstanceIds=instance_ids)
        logger.info(f"Terminated instances: {instance_ids}")
        return response

    def get_instance_status(self, instance_ids: List[str]) -> List[Dict[str, Any]]:
        """Get detailed instance status."""
        ec2 = self._get_client("ec2")
        response = ec2.describe_instance_status(InstanceIds=instance_ids)
        return response.get("InstanceStatuses", [])

    # ========== IAM Management ==========

    def list_users(self, path_prefix: str = "/") -> List[IAMUser]:
        """List IAM users."""
        iam = self._get_client("iam")
        response = iam.list_users(PathPrefix=path_prefix)

        users = []
        for u in response.get("Users", []):
            users.append(IAMUser(
                user_name=u.get("UserName", ""),
                user_id=u.get("UserId", ""),
                arn=u.get("Arn", ""),
                create_date=u.get("CreateDate"),
                password_last_used=u.get("PasswordLastUsed"),
                path=u.get("Path", "/"),
            ))

        return users

    def create_user(
        self,
        user_name: str,
        path: str = "/",
        tags: Dict[str, str] = None,
    ) -> IAMUser:
        """Create an IAM user."""
        iam = self._get_client("iam")

        params = {
            "UserName": user_name,
            "Path": path,
        }
        if tags:
            params["Tags"] = [{"Key": k, "Value": v} for k, v in tags.items()]

        response = iam.create_user(**params)
        user_data = response.get("User", {})

        logger.info(f"Created IAM user: {user_name}")

        return IAMUser(
            user_name=user_data.get("UserName", ""),
            user_id=user_data.get("UserId", ""),
            arn=user_data.get("Arn", ""),
            path=user_data.get("Path", "/"),
        )

    def delete_user(self, user_name: str) -> bool:
        """Delete an IAM user."""
        iam = self._get_client("iam")
        iam.delete_user(UserName=user_name)
        logger.info(f"Deleted IAM user: {user_name}")
        return True

    def create_access_key(self, user_name: str) -> Dict[str, str]:
        """Create access key for user."""
        iam = self._get_client("iam")
        response = iam.create_access_key(UserName=user_name)

        key_data = response.get("AccessKey", {})
        logger.info(f"Created access key for user: {user_name}")

        return {
            "access_key_id": key_data.get("AccessKeyId", ""),
            "secret_access_key": key_data.get("SecretAccessKey", ""),
        }

    def list_groups(self, path_prefix: str = "/") -> List[Dict[str, Any]]:
        """List IAM groups."""
        iam = self._get_client("iam")
        response = iam.list_groups(PathPrefix=path_prefix)
        return response.get("Groups", [])

    def add_user_to_group(self, user_name: str, group_name: str) -> bool:
        """Add user to group."""
        iam = self._get_client("iam")
        iam.add_user_to_group(UserName=user_name, GroupName=group_name)
        logger.info(f"Added {user_name} to group {group_name}")
        return True

    def remove_user_from_group(self, user_name: str, group_name: str) -> bool:
        """Remove user from group."""
        iam = self._get_client("iam")
        iam.remove_user_from_group(UserName=user_name, GroupName=group_name)
        logger.info(f"Removed {user_name} from group {group_name}")
        return True

    # ========== S3 Management ==========

    def list_buckets(self) -> List[S3Bucket]:
        """List S3 buckets."""
        s3 = self._get_client("s3")
        response = s3.list_buckets()

        buckets = []
        for b in response.get("Buckets", []):
            buckets.append(S3Bucket(
                name=b.get("Name", ""),
                creation_date=b.get("CreationDate"),
            ))

        return buckets

    def create_bucket(
        self,
        bucket_name: str,
        region: str = None,
    ) -> bool:
        """Create an S3 bucket."""
        s3 = self._get_client("s3")

        params = {"Bucket": bucket_name}
        region = region or self.config.region

        if region != "us-east-1":
            params["CreateBucketConfiguration"] = {"LocationConstraint": region}

        s3.create_bucket(**params)
        logger.info(f"Created S3 bucket: {bucket_name}")
        return True

    def delete_bucket(self, bucket_name: str) -> bool:
        """Delete an S3 bucket."""
        s3 = self._get_client("s3")
        s3.delete_bucket(Bucket=bucket_name)
        logger.info(f"Deleted S3 bucket: {bucket_name}")
        return True

    def list_objects(
        self,
        bucket_name: str,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List objects in bucket."""
        s3 = self._get_client("s3")
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=max_keys,
        )
        return response.get("Contents", [])

    # ========== VPC & Networking ==========

    def list_vpcs(self) -> List[Dict[str, Any]]:
        """List VPCs."""
        ec2 = self._get_client("ec2")
        response = ec2.describe_vpcs()
        return response.get("Vpcs", [])

    def list_subnets(self, vpc_id: str = None) -> List[Dict[str, Any]]:
        """List subnets."""
        ec2 = self._get_client("ec2")

        params = {}
        if vpc_id:
            params["Filters"] = [{"Name": "vpc-id", "Values": [vpc_id]}]

        response = ec2.describe_subnets(**params)
        return response.get("Subnets", [])

    def list_security_groups(self, vpc_id: str = None) -> List[Dict[str, Any]]:
        """List security groups."""
        ec2 = self._get_client("ec2")

        params = {}
        if vpc_id:
            params["Filters"] = [{"Name": "vpc-id", "Values": [vpc_id]}]

        response = ec2.describe_security_groups(**params)
        return response.get("SecurityGroups", [])

    # ========== CloudWatch ==========

    def get_metrics(
        self,
        namespace: str,
        metric_name: str,
        dimensions: List[Dict] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        period: int = 300,
        statistic: str = "Average",
    ) -> List[Dict[str, Any]]:
        """Get CloudWatch metrics."""
        cloudwatch = self._get_client("cloudwatch")

        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(hours=1))

        params = {
            "Namespace": namespace,
            "MetricName": metric_name,
            "StartTime": start_time,
            "EndTime": end_time,
            "Period": period,
            "Statistics": [statistic],
        }

        if dimensions:
            params["Dimensions"] = dimensions

        response = cloudwatch.get_metric_statistics(**params)
        return response.get("Datapoints", [])

    def list_alarms(self, state: str = None) -> List[Dict[str, Any]]:
        """List CloudWatch alarms."""
        cloudwatch = self._get_client("cloudwatch")

        params = {}
        if state:
            params["StateValue"] = state

        response = cloudwatch.describe_alarms(**params)
        return response.get("MetricAlarms", [])

    # ========== Cost Explorer ==========

    def get_cost_and_usage(
        self,
        start_date: str,
        end_date: str,
        granularity: str = "DAILY",
        metrics: List[str] = None,
        group_by: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Get cost and usage data."""
        ce = self._get_client("ce")

        params = {
            "TimePeriod": {
                "Start": start_date,
                "End": end_date,
            },
            "Granularity": granularity,
            "Metrics": metrics or ["UnblendedCost"],
        }

        if group_by:
            params["GroupBy"] = group_by

        response = ce.get_cost_and_usage(**params)
        return response

    # ========== Security Hub ==========

    def get_findings(
        self,
        filters: Dict = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get Security Hub findings."""
        try:
            securityhub = self._get_client("securityhub")

            params = {"MaxResults": max_results}
            if filters:
                params["Filters"] = filters

            response = securityhub.get_findings(**params)
            return response.get("Findings", [])

        except Exception as e:
            logger.error(f"Security Hub error: {e}")
            return []

    @staticmethod
    def get_setup_guide() -> str:
        """Get AWS setup instructions."""
        return """
# AWS Integration Setup Guide

## Prerequisites
- AWS account
- IAM user or role with appropriate permissions
- AWS CLI configured (optional, for profile auth)

## Step 1: Create IAM User for cosmos

1. Go to AWS Console > IAM > Users
2. Click "Add users"
3. User name: "cosmos-manager"
4. Select "Access key - Programmatic access"
5. Click "Next: Permissions"

## Step 2: Attach Policies

For comprehensive management, attach these policies:
- `AmazonEC2FullAccess` (EC2 management)
- `IAMFullAccess` (IAM management)
- `AmazonS3FullAccess` (S3 management)
- `AmazonVPCFullAccess` (VPC management)
- `CloudWatchFullAccess` (Monitoring)
- `AWSBillingReadOnlyAccess` (Cost)
- `AWSSecurityHubReadOnlyAccess` (Security)

Or create a custom policy with least-privilege.

## Step 3: Create Access Keys

1. Select your user in IAM
2. Go to "Security credentials" tab
3. Click "Create access key"
4. Select "Application running outside AWS"
5. Download the CSV with your keys

## Step 4: Configure cosmos

### Option A: Environment Variables

```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=us-east-1
```

### Option B: AWS Profile

Create `~/.aws/credentials`:
```ini
[cosmos]
aws_access_key_id = your-access-key
aws_secret_access_key = your-secret-key
```

Create `~/.aws/config`:
```ini
[profile cosmos]
region = us-east-1
output = json
```

### Option C: Direct Configuration

```python
from Cosmos.integration.cloud.aws_manager import AWSManager, AWSConfig

config = AWSConfig(
    access_key_id="your-access-key",
    secret_access_key="your-secret-key",
    region="us-east-1",
)

manager = AWSManager(config)
```

## Step 5: Test Connection

```python
manager = AWSManager(config)
if manager.authenticate():
    instances = manager.list_instances()
    print(f"Found {len(instances)} EC2 instances")
```

## Using AWS Profiles

```python
config = AWSConfig(
    auth_method=AWSAuthMethod.PROFILE,
    profile="cosmos",
    region="us-east-1",
)
manager = AWSManager(config)
```

## Using Instance Role (When Running in AWS)

```python
config = AWSConfig(
    auth_method=AWSAuthMethod.INSTANCE_ROLE,
    region="us-east-1",
)
manager = AWSManager(config)
```

## Security Best Practices

1. Use least-privilege IAM policies
2. Enable MFA for the IAM user
3. Use IAM roles instead of keys when possible
4. Rotate access keys regularly
5. Never commit keys to version control
6. Use AWS Organizations for multi-account
7. Enable CloudTrail for auditing
"""


# Global instance (requires configuration)
aws_manager = AWSManager()
