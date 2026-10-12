# Advanced Validation Patterns

Complete guide to validation in Instructor using Pydantic.

## Table of Contents
- Built-in Validators
- Custom Field Validators
- Model-Level Validation
- Complex Validation Patterns
- Error Handling

## Built-in Validators

### Numeric Constraints

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    price: float = Field(gt=0, description="Price must be positive")
    discount: float = Field(ge=0, le=100, description="Discount 0-100%")
    quantity: int = Field(ge=1, description="At least 1 item")
    rating: float = Field(ge=0.0, le=5.0, description="Rating 0-5 stars")

# If LLM provides invalid values, automatic retry with error feedback
```

**Available constraints:**
- `gt`: Greater than
- `ge`: Greater than or equal
- `lt`: Less than
- `le`: Less than or equal
- `multiple_of`: Must be multiple of this number

### String Constraints

```python
class User(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=20,
        pattern=r'^[a-zA-Z0-9_]+$',
        description="3-20 alphanumeric characters"
    )
    bio: str = Field(max_length=500, description="Bio up to 500 chars")
    status: str = Field(pattern=r'^(active|inactive|pending)$')

# pattern validates against regex
```

### Email and URL Validation

```python
from pydantic import EmailStr, HttpUrl, AnyUrl

class Contact(BaseModel):
    email: EmailStr  # Validates email format
    website: HttpUrl  # Validates HTTP/HTTPS URLs
    portfolio: AnyUrl  # Any valid URL scheme

contact = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Extract: john@example.com, https://example.com"
    }],
    response_model=Contact
)
```

### Date and DateTime Validation

```python
from datetime import date, datetime
from pydantic import Field, field_validator

class Event(BaseModel):
    event_date: date  # Validates date format
    created_at: datetime  # Validates datetime format
    year: int = Field(ge=1900, le=2100)

    @field_validator('event_date')
    def future_date(cls, v):
        """Ensure event is in the future."""
        if v < date.today():
            raise ValueError('Event must be in the future')
        return v
```

### List and Dict Validation

```python
class Document(BaseModel):
    tags: list[str] = Field(min_length=1, max_length=10)
    keywords: list[str] = Field(min_length=3, description="At least 3 keywords")
    metadata: dict[str, str] = Field(description="String key-value pairs")

    @field_validator('tags')
    def unique_tags(cls, v):
        """Ensure tags are unique."""
        if len(v) != len(set(v)):
            raise ValueError('Tags must be unique')
        return v
```

## Custom Field Validators

### Basic Field Validator

```python
from pydantic import field_validator

class Person(BaseModel):
    name: str
    age: int

    @field_validator('name')
    def name_must_not_be_empty(cls, v):
        """Validate name is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

    @field_validator('age')
    def age_must_be_reasonable(cls, v):
        """Validate age is between 0 and 120."""
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v
```

### Validator with Field Info

```python
from pydantic import ValidationInfo

class Article(BaseModel):
    title: str
    content: str

    @field_validator('content')
    def content_length(cls, v, info: ValidationInfo):
        """Validate content is longer than title."""
        if 'title' in info.data:
            title_len = len(info.data['title'])
            if len(v) < title_len * 2:
                raise ValueError('Content should be at least 2x title length')
        return v
```

### Multiple Fields Validation

```python
class TimeRange(BaseModel):
    start_time: str
    end_time: str

    @field_validator('start_time', 'end_time')
    def valid_time_format(cls, v):
        """Validate both times are in HH:MM format."""
        import re
        if not re.match(r'^\d{2}:\d{2}$', v):
            raise ValueError('Time must be in HH:MM format')
        return v
```

### Transform and Validate

```python
class URL(BaseModel):
    url: str

    @field_validator('url')
    def normalize_url(cls, v):
        """Add https:// if missing."""
        if not v.startswith(('http://', 'https://')):
            v = f'https://{v}'
        return v
```

## Model-Level Validation

### Cross-Field Validation

```python
from pydantic import model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode='after')
    def check_dates(self):
        """Ensure end_date is after start_date."""
        from datetime import datetime
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        if end < start:
            raise ValueError('end_date must be after start_date')
        return self

class PriceRange(BaseModel):
    min_price: float
    max_price: float

    @model_validator(mode='after')
    def check_price_range(self):
        """Ensure max > min."""
        if self.max_price <= self.min_price:
            raise ValueError('max_price must be greater than min_price')
        return self
```

### Conditional Validation

```python
class Order(BaseModel):
    order_type: str  # "standard" or "express"
    delivery_date: str
    delivery_time: Optional[str] = None

    @model_validator(mode='after')
    def check_delivery_time(self):
        """Express orders need delivery time."""
        if self.order_type == "express" and not self.delivery_time:
            raise ValueError('Express orders require delivery_time')
        return self
```

### Complex Business Logic

```python
class Discount(BaseModel):
    code: str
    percentage: float = Field(ge=0, le=100)
    min_purchase: float = Field(ge=0)
    max_discount: float = Field(ge=0)

    @model_validator(mode='after')
    def validate_discount(self):
        """Ensure discount logic is sound."""
        # Max discount can't exceed percentage of min_purchase
        theoretical_max = (self.percentage / 100) * self.min_purchase
        if self.max_discount > theoretical_max:
            self.max_discount = theoretical_max
        return self
```

## Complex Validation Patterns

### Nested Model Validation

```python
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

    @field_validator('postal_code')
    def validate_postal_code(cls, v, info: ValidationInfo):
        """Validate postal code format based on country."""
        if 'country' in info.data:
            country = info.data['country']
            if country == "USA":
                import re
                if not re.match(r'^\d{5}(-\d{4})?$', v):
                    raise ValueError('Invalid US postal code')
            elif country == "Canada":
                if not re.match(r'^[A-Z]\d[A-Z] \d[A-Z]\d$', v):
                    raise ValueError('Invalid Canadian postal code')
        return v

class Person(BaseModel):
    name: str
    address: Address

# Nested validation runs automatically
```

### List of Models

```python
class Task(BaseModel):
    title: str = Field(min_length=1)
    priority: int = Field(ge=1, le=5)

class Project(BaseModel):
    name: str
    tasks: list[Task] = Field(min_length=1, description="At least 1 task")

    @field_validator('tasks')
    def at_least_one_high_priority(cls, v):
        """Ensure at least one task has priority >= 4."""
        if not any(task.priority >= 4 for task in v):
            raise ValueError('Project needs at least one high-priority task')
        return v
```

### Union Type Validation

```python
from typing import Union

class TextBlock(BaseModel):
    type: str = "text"
    content: str = Field(min_length=1)

class ImageBlock(BaseModel):
    type: str = "image"
    url: HttpUrl
    alt_text: str

class Page(BaseModel):
    title: str
    blocks: list[Union[TextBlock, ImageBlock]]

    @field_validator('blocks')
    def validate_block_types(cls, v):
        """Ensure first block is TextBlock."""
        if v and not isinstance(v[0], TextBlock):
            raise ValueError('First block must be text')
        return v
```

### Dependent Fields

```python
class Subscription(BaseModel):
    plan: str  # "free", "pro", "enterprise"
    max_users: int
    features: list[str]

    @model_validator(mode='after')
    def validate_plan_limits(self):
        """Enforce plan-specific limits."""
        limits = {
            "free": {"max_users": 1, "required_features": ["basic"]},
            "pro": {"max_users": 10, "required_features": ["basic", "advanced"]},
            "enterprise": {"max_users": 999, "required_features": ["basic", "advanced", "premium"]}
        }

        if self.plan in limits:
            limit = limits[self.plan]

            if self.max_users > limit["max_users"]:
                raise ValueError(f'{self.plan} plan limited to {limit["max_users"]} users')

            for feature in limit["required_features"]:
                if feature not in self.features:
                    raise ValueError(f'{self.plan} plan requires {feature} feature')

        return self
```

## Error Handling

### Graceful Degradation

```python
class OptionalExtraction(BaseModel):
    # Required fields
    title: str

    # Optional fields with defaults
    author: Optional[str] = None
    date: Optional[str] = None
    tags: list[str] = Field(default_factory=list)

# LLM can succeed even if it can't extract everything
```

### Partial Validation

```python
from pydantic import ValidationError

def extract_with_fallback(text: str):
    """Try full extraction, fall back to partial."""
    try:
        # Try full extraction
        return client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": text}],
            response_model=FullModel
        )
    except ValidationError:
        # Fall back to partial model
        return client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": text}],
            response_model=PartialModel
        )
```

### Validation Error Inspection

```python
from pydantic import ValidationError

try:
    result = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[...],
        response_model=MyModel,
        max_retries=3
    )
except ValidationError as e:
    # Inspect specific errors
    for error in e.errors():
        field = error['loc'][0]
        message = error['msg']
        print(f"Field '{field}' failed: {message}")

        # Custom handling per field
        if field == 'email':
            # Handle email validation failure
            pass
```

### Custom Error Messages

```python
class DetailedModel(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=100,
        description="Name between 2-100 characters"
    )
    age: int = Field(
        ge=0,
        le=120,
        description="Age between 0 and 120 years"
    )

    @field_validator('name')
    def validate_name(cls, v):
        """Provide helpful error message."""
        if not v.strip():
            raise ValueError(
                'Name cannot be empty. '
                'Please provide a valid name from the text.'
            )
        return v

# When validation fails, LLM sees these helpful messages
```

## Validation Best Practices

### 1. Be Specific

```python
# ❌ Bad: Vague validation
class Item(BaseModel):
    name: str

# ✅ Good: Specific constraints
class Item(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=200,
        description="Item name, 1-200 characters"
    )
```

### 2. Provide Context

```python
# ✅ Good: Explain why validation failed
@field_validator('price')
def validate_price(cls, v):
    if v <= 0:
        raise ValueError(
            'Price must be positive. '
            'Extract numeric price from text without currency symbols.'
        )
    return v
```

### 3. Use Enums for Fixed Sets

```python
# ❌ Bad: String validation
status: str

@field_validator('status')
def validate_status(cls, v):
    if v not in ['active', 'inactive', 'pending']:
        raise ValueError('Invalid status')
    return v

# ✅ Good: Enum
class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

status: Status  # Validation automatic
```

### 4. Balance Strictness

```python
# Too strict: May fail unnecessarily
class StrictModel(BaseModel):
    date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')
    # Fails if LLM uses "2024-1-5" instead of "2024-01-05"

# Better: Normalize in validator
class FlexibleModel(BaseModel):
    date: str

    @field_validator('date')
    def normalize_date(cls, v):
        from datetime import datetime
        # Parse flexible formats
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y']:
            try:
                dt = datetime.strptime(v, fmt)
                return dt.strftime('%Y-%m-%d')  # Normalize
            except ValueError:
                continue
        raise ValueError('Invalid date format')
```

### 5. Test Validation

```python
# Test your validators with edge cases
def test_validation():
    # Should succeed
    valid = MyModel(field="valid_value")

    # Should fail
    try:
        invalid = MyModel(field="invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected

# Run tests before using in production
```

## Advanced Techniques

### Conditional Required Fields

```python
from typing import Optional

class ConditionalModel(BaseModel):
    type: str
    detail_a: Optional[str] = None
    detail_b: Optional[str] = None

    @model_validator(mode='after')
    def check_required_details(self):
        """Require different fields based on type."""
        if self.type == "type_a" and not self.detail_a:
            raise ValueError('type_a requires detail_a')
        if self.type == "type_b" and not self.detail_b:
            raise ValueError('type_b requires detail_b')
        return self
```

### Validation with External Data

```python
class Product(BaseModel):
    sku: str
    name: str

    @field_validator('sku')
    def validate_sku(cls, v):
        """Check SKU exists in database."""
        # Query database or API
        if not database.sku_exists(v):
            raise ValueError(f'SKU {v} not found in catalog')
        return v
```

### Progressive Validation

```python
# Start with loose validation
class Stage1(BaseModel):
    data: str  # Any string

# Then strict validation
class Stage2(BaseModel):
    data: str = Field(pattern=r'^[A-Z]{3}-\d{6}$')

# Use Stage1 for initial extraction
# Use Stage2 for final validation
```

## Resources

- **Pydantic Docs**: https://docs.pydantic.dev/latest/concepts/validators/
- **Instructor Examples**: https://python.useinstructor.com/examples
